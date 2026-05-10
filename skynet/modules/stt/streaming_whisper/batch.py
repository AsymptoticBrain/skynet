import io
import math
from typing import Any

import httpx

import skynet.modules.stt.streaming_whisper.cfg as cfg
from skynet.env import (
    whisper_backend,
    whisper_beam_size,
    whisper_remote_model,
)
from skynet.logs import get_logger
from skynet.modules.stt.streaming_whisper.utils.utils import (
    WhisperResult,
    WhisperSegment,
    WhisperWord,
    _get_remote_client,
)

log = get_logger(__name__)


def _result_to_verbose_json(result: WhisperResult, duration: float | None) -> dict[str, Any]:
    """Render a WhisperResult in the OpenAI verbose_json shape."""

    def seg_dump(seg: WhisperSegment) -> dict[str, Any]:
        return {
            'id': seg.id,
            'seek': seg.seek,
            'start': seg.start,
            'end': seg.end,
            'text': seg.text,
            'tokens': seg.tokens,
            'temperature': seg.temperature,
            'avg_logprob': seg.avg_logprob,
            'compression_ratio': seg.compression_ratio,
            'no_speech_prob': seg.no_speech_prob,
        }

    def word_dump(w: WhisperWord) -> dict[str, Any]:
        return {'word': w.word, 'start': w.start, 'end': w.end}

    payload: dict[str, Any] = {
        'task': 'transcribe',
        'language': result.language,
        'text': result.text,
        'segments': [seg_dump(s) for s in result.segments],
        'words': [word_dump(w) for w in result.words],
    }
    if duration is not None and not math.isnan(duration):
        payload['duration'] = duration
    return payload


def _transcribe_local_file(file_bytes: bytes, lang: str | None, prompt: str | None) -> tuple[WhisperResult, float]:
    """Run faster-whisper on an arbitrary container/codec by handing the bytes
    to PyAV via a BytesIO buffer."""
    buffer = io.BytesIO(file_bytes)
    iterator, info = cfg.model.transcribe(
        buffer,
        language=lang,
        task='transcribe',
        word_timestamps=True,
        beam_size=whisper_beam_size,
        initial_prompt=prompt or None,
        condition_on_previous_text=False,
    )
    segments = list(iterator)
    result = WhisperResult.from_faster_whisper(segments)
    if not result.language:
        result.language = getattr(info, 'language', '') or (lang or '')
    duration = float(getattr(info, 'duration', 0.0) or 0.0)
    return result, duration


def _transcribe_remote_file(
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    lang: str | None,
    prompt: str | None,
) -> tuple[WhisperResult, float]:
    data: dict[str, str] = {
        'model': whisper_remote_model,
        'response_format': 'verbose_json',
        'timestamp_granularities[]': 'word',
        'temperature': '0',
    }
    if lang:
        data['language'] = lang
    if prompt and prompt.strip():
        data['prompt'] = prompt.strip()

    files = {'file': (filename, file_bytes, content_type or 'application/octet-stream')}

    client = _get_remote_client()
    try:
        response = client.post('/v1/audio/transcriptions', data=data, files=files)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f'Remote whisper request failed: {e}') from e

    try:
        payload = response.json()
    except ValueError as e:
        raise RuntimeError(f'Remote whisper returned non-JSON response: {e}') from e

    result = WhisperResult.from_verbose_json(payload)
    duration = float(payload.get('duration') or 0.0)
    return result, duration


def transcribe_file(
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    lang: str | None,
    prompt: str | None,
) -> tuple[WhisperResult, float]:
    """Transcribe a complete audio file. Picks the backend Skynet is configured for.

    Returns (result, duration_seconds).
    """
    if whisper_backend == 'remote':
        return _transcribe_remote_file(file_bytes, filename, content_type, lang, prompt)
    return _transcribe_local_file(file_bytes, lang, prompt)


def render_response(
    result: WhisperResult,
    duration: float | None,
    response_format: str,
) -> dict[str, Any] | str:
    """Return the response body in the OpenAI-compatible shape requested.

    Supported: 'json', 'text', 'verbose_json'. Other values fall back to
    'verbose_json' to preserve the most information.
    """
    fmt = (response_format or 'verbose_json').lower().strip()
    if fmt == 'text':
        return result.text
    if fmt == 'json':
        return {'text': result.text}
    return _result_to_verbose_json(result, duration)
