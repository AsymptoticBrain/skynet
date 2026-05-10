import asyncio

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)

from skynet.auth.bearer import JWTBearer
from skynet.env import bypass_auth, whisper_batch_max_upload_bytes
from skynet.logs import get_logger
from skynet.modules.stt.streaming_whisper.batch import render_response, transcribe_file
from skynet.modules.stt.streaming_whisper.connection_manager import ConnectionManager
from skynet.modules.stt.streaming_whisper.utils import utils

log = get_logger(__name__)

ws_connection_manager = ConnectionManager()
app = FastAPI()  # No need for CORS middleware

_http_dependencies = [] if bypass_auth else [Depends(JWTBearer())]


@app.websocket('/ws/{meeting_id}')
async def websocket_endpoint(websocket: WebSocket, meeting_id: str, auth_token: str | None = None):
    connection = await ws_connection_manager.connect(websocket, meeting_id, auth_token)
    if connection:
        while connection.connected:
            try:
                chunk = await websocket.receive_bytes()
            except WebSocketDisconnect:
                log.info(f'Meeting {connection.meeting_id} has ended')
                await ws_connection_manager.disconnect(connection, already_closed=True)
                break
            except WebSocketException as wserr:
                log.warning(f'Error on websocket {connection.meeting_id}. Error {wserr.__class__}: \n{wserr}')
                await ws_connection_manager.disconnect(connection)
                break
            except Exception as err:
                log.warning(
                    f'Expected bytes, received something else, disconnecting {connection.meeting_id}. Error {err.__class__}: \n{err}'
                )
                await ws_connection_manager.disconnect(connection)
                break
            if len(chunk) == 1 and ord(b'' + chunk) == 0:
                log.info(f'Received disconnect message for {connection.meeting_id}')
                await ws_connection_manager.disconnect(connection)
                break
            await ws_connection_manager.process(connection, chunk, utils.now())


_UPLOAD_READ_CHUNK = 1024 * 1024


@app.post('/v1/audio/transcriptions', dependencies=_http_dependencies)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),  # accepted for OpenAI-API compatibility, ignored
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form('verbose_json'),
    temperature: float | None = Form(None),  # accepted for compatibility, ignored
):
    """OpenAI-compatible batch transcription. Accepts a complete audio file
    (any container/codec ffmpeg/PyAV can decode for the local backend, anything
    the upstream server accepts for the remote backend) and returns the
    transcript synchronously."""

    too_large = HTTPException(
        status_code=413,
        detail=f'Upload exceeds WHISPER_BATCH_MAX_UPLOAD_BYTES ({whisper_batch_max_upload_bytes} bytes)',
    )

    # Fast path: trust Content-Length when present so a client streaming a
    # multi-GB upload gets rejected before we buffer anything. The body is
    # multipart so this is an upper bound on the file payload, which is fine
    # for a reject-only check.
    content_length = request.headers.get('content-length')
    if content_length:
        try:
            if int(content_length) > whisper_batch_max_upload_bytes:
                raise too_large
        except ValueError:
            pass

    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > whisper_batch_max_upload_bytes:
            raise too_large
        chunks.append(chunk)

    if size == 0:
        raise HTTPException(status_code=400, detail='Empty upload')

    file_bytes = b''.join(chunks)
    chunks.clear()

    log.info(
        f'Batch transcription: filename={file.filename!r} size={size} ' f'lang={language!r} fmt={response_format!r}'
    )

    try:
        result, duration = await asyncio.to_thread(
            transcribe_file,
            file_bytes,
            file.filename or 'audio',
            file.content_type,
            language,
            prompt,
        )
    except RuntimeError as e:
        log.warning(f'Batch transcription failed: {e}')
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        log.exception('Batch transcription crashed')
        raise HTTPException(status_code=500, detail=f'Transcription failed: {e}') from e

    body = render_response(result, duration, response_format)
    if isinstance(body, str):
        return Response(content=body, media_type='text/plain')
    return body
