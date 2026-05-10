import asyncio

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
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


# Headroom over WHISPER_BATCH_MAX_UPLOAD_BYTES for multipart envelope (boundaries
# + the small auxiliary form fields). Anything past this is rejected by Content-
# Length before Starlette parses the body, so a multi-GB POST never gets spooled.
_MULTIPART_OVERHEAD_HEADROOM = 64 * 1024


class _BatchUploadSizeMiddleware:
    """ASGI middleware that 413s oversized batch transcription uploads on the
    Content-Length header alone, before FastAPI's `File(...)` dependency
    triggers multipart parsing and spools the entire body."""

    def __init__(self, asgi_app, path: str, max_bytes: int):
        self._app = asgi_app
        self._path = path
        self._limit = max_bytes + _MULTIPART_OVERHEAD_HEADROOM

    async def __call__(self, scope, receive, send):
        if (
            scope.get('type') == 'http'
            and scope.get('method') == 'POST'
            and scope.get('path') == self._path
        ):
            for name, value in scope.get('headers', ()):
                if name == b'content-length':
                    try:
                        declared = int(value)
                    except ValueError:
                        break
                    if declared > self._limit:
                        await send(
                            {
                                'type': 'http.response.start',
                                'status': 413,
                                'headers': [(b'content-type', b'application/json')],
                            }
                        )
                        await send(
                            {
                                'type': 'http.response.body',
                                'body': (
                                    b'{"detail":"Upload exceeds '
                                    b'WHISPER_BATCH_MAX_UPLOAD_BYTES"}'
                                ),
                            }
                        )
                        return
                    break
        await self._app(scope, receive, send)


app.add_middleware(
    _BatchUploadSizeMiddleware,
    path='/v1/audio/transcriptions',
    max_bytes=whisper_batch_max_upload_bytes,
)


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

    # Stream the upload from the spooled multipart part and abort the moment
    # the audio payload itself exceeds the cap. The middleware above already
    # rejected truly huge requests on Content-Length; this catches the case
    # where Content-Length was missing or understated.
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > whisper_batch_max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f'Upload exceeds WHISPER_BATCH_MAX_UPLOAD_BYTES '
                    f'({whisper_batch_max_upload_bytes} bytes)'
                ),
            )
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
