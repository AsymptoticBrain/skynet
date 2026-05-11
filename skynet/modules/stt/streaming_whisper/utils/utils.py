import math
import secrets
import time
from datetime import datetime, timezone
from typing import List, Tuple

import httpx
import numpy as np
from numpy import ndarray
from pydantic import BaseModel
from silero_vad import get_speech_timestamps, read_audio
from uuid6 import UUID

import skynet.modules.stt.streaming_whisper.cfg as cfg
from skynet.env import (
    whisper_backend,
    whisper_beam_size,
    whisper_min_probability,
    whisper_remote_api_key,
    whisper_remote_model,
    whisper_remote_timeout,
    whisper_remote_url,
)
from skynet.logs import get_logger

log = get_logger(__name__)


class WhisperWord(BaseModel):
    probability: float
    word: str
    start: float
    end: float


class WhisperSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List


class TranscriptionResponse(BaseModel):
    id: str
    participant_id: str
    ts: int
    text: str
    audio: str
    type: str
    variance: float


class CutMark(BaseModel):
    start: float = 0.0
    end: float = 0.0
    probability: float = 0.0


class WhisperResult:
    text: str
    segments: list[WhisperSegment]
    words: list[WhisperWord]
    confidence: float
    language: str

    def __init__(
        self,
        text: str = '',
        segments: list[WhisperSegment] | None = None,
        words: list[WhisperWord] | None = None,
        language: str = '',
    ):
        self.text = text
        self.segments = segments or []
        self.words = words or []
        self.language = language
        self.confidence = self.get_confidence()

    @classmethod
    def from_faster_whisper(cls, ts_result) -> 'WhisperResult':
        return cls(
            text=''.join([segment.text for segment in ts_result]),
            segments=[WhisperSegment.model_validate(segment._asdict()) for segment in ts_result],
            words=[WhisperWord.model_validate(word._asdict()) for segment in ts_result for word in segment.words],
        )

    @classmethod
    def from_verbose_json(cls, data: dict) -> 'WhisperResult':
        """Build a WhisperResult from an OpenAI-compatible verbose_json response.

        Word-level probability isn't exposed in the OpenAI format, so we derive
        a per-word probability from the enclosing segment's ``avg_logprob``
        (falling back to 1.0 when absent) to keep cut-mark heuristics working.
        """
        seg_data = data.get('segments') or []
        word_data = data.get('words') or []

        def seg_prob(seg: dict) -> float:
            avg = seg.get('avg_logprob')
            if avg is None:
                return 1.0
            try:
                return float(math.exp(avg))
            except (ValueError, OverflowError):
                return 1.0

        segments = [
            WhisperSegment(
                id=int(seg.get('id', i)),
                seek=int(seg.get('seek', 0)),
                start=float(seg.get('start', 0.0)),
                end=float(seg.get('end', 0.0)),
                text=seg.get('text', ''),
                tokens=list(seg.get('tokens', [])),
                temperature=float(seg.get('temperature', 0.0)),
                avg_logprob=float(seg.get('avg_logprob', 0.0)),
                compression_ratio=float(seg.get('compression_ratio', 0.0)),
                no_speech_prob=float(seg.get('no_speech_prob', 0.0)),
                words=[],
            )
            for i, seg in enumerate(seg_data)
        ]

        words: list[WhisperWord] = []
        for w in word_data:
            w_start = float(w.get('start', 0.0))
            w_end = float(w.get('end', w_start))
            prob = 1.0
            for raw_seg in seg_data:
                s_start = float(raw_seg.get('start', 0.0))
                s_end = float(raw_seg.get('end', float('inf')))
                if s_start <= w_start <= s_end:
                    prob = seg_prob(raw_seg)
                    break
            words.append(
                WhisperWord(
                    word=w.get('word', ''),
                    start=w_start,
                    end=w_end,
                    probability=prob,
                )
            )

        text = data.get('text', '') or ''.join(seg.text for seg in segments)

        # Fallback: if we got text and segments but no word timestamps (some
        # servers omit them), synthesize a single word so the downstream
        # cut-mark logic has something to work with.
        if not words and segments:
            words = [
                WhisperWord(
                    word=text.strip(),
                    start=segments[0].start,
                    end=segments[-1].end,
                    probability=seg_prob(seg_data[0]) if seg_data else 1.0,
                )
            ]

        return cls(text=text, segments=segments, words=words, language=data.get('language', ''))

    def __str__(self):
        return (
            f'Text: {self.text}\n'
            + f'Confidence avg: {self.confidence}\n'
            + f'Segments: {self.segments}\n'
            + f'Words: {self.words}'
        )

    def get_confidence(self) -> float:
        if len(self.words) > 0:
            return float(sum(word.probability for word in self.words) / len(self.words))
        return 0.0


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sudanese",
}

# List of final transcriptions which should not be included in the initial prompt.
# This is to prevent the model from repeating the same text over and over or become
# biased towards a specific way of transcribing.
black_listed_prompts = ['. .']


def convert_bytes_to_seconds(byte_str: bytes) -> float:
    return round(len(byte_str) * cfg.one_byte_s, 3)


def convert_seconds_to_bytes(cut_mark: float) -> int:
    return int(cut_mark / cfg.one_byte_s)


def is_silent(audio: bytes) -> Tuple[bool, iter]:
    chunk_duration = convert_bytes_to_seconds(audio)
    wav_header = get_wav_header([audio], chunk_duration_s=chunk_duration)
    stream = wav_header + b'' + audio
    audio = read_audio(stream)
    st = get_speech_timestamps(audio, model=cfg.vad_model, return_seconds=True)
    log.debug(f'Detected speech timestamps: {st}')
    silent = True if len(st) == 0 else False
    return silent, st


def get_phrase_prob(last_word_idx: int, words: list[WhisperWord]) -> float:
    word_number = last_word_idx + 1
    return sum([word.probability for word in words[:word_number]]) / word_number


def find_biggest_gap_between_words(word_list: list[WhisperWord]) -> CutMark:
    prev_word = word_list[0]
    biggest_gap_so_far = 0.0
    result = CutMark()
    for i, word in enumerate(word_list):
        if i == 0:
            continue
        diff = word.start - prev_word.end
        probability = get_phrase_prob(i - 1, word_list)
        if diff > biggest_gap_so_far:
            biggest_gap_so_far = diff
            result = CutMark(start=prev_word.end, end=word.start, probability=probability)
            log.debug(f'Biggest gap between words:\n{result}')
        prev_word = word
    return result


def get_cut_mark_from_segment_probability(ts_result: WhisperResult) -> CutMark:
    check_len = len(ts_result.words) - 1
    phrase = ''
    if len(ts_result.words) > 1:
        # force a final at the biggest gap between words found if the audio is longer than 10 seconds
        if ts_result.words[-1].end >= 10:
            return find_biggest_gap_between_words(ts_result.words)
        for i, word in enumerate(ts_result.words):
            if i == check_len:
                break
            phrase += word.word
            avg_probability = get_phrase_prob(i, ts_result.words)
            if len(phrase) >= 48:
                if (
                    avg_probability >= whisper_min_probability
                    and word.word[-1] in ['.', '!', '?']
                    and word.end < ts_result.words[i + 1].start
                ):
                    log.debug(f'Found split at {word.word} ({word.end} - {ts_result.words[i+1].start})')
                    log.debug(f'Avg probability: {avg_probability}')
                    return CutMark(start=word.end, end=ts_result.words[i + 1].start, probability=avg_probability)
                else:
                    if ts_result.words[-1].end >= 15:
                        return find_biggest_gap_between_words(ts_result.words)
    return CutMark()


def get_wav_header(chunks: List[bytes], chunk_duration_s: float = 0.256, sample_rate: int = 16000) -> bytes:
    duration = len(chunks) * chunk_duration_s
    samples = int(duration * sample_rate)
    bits_per_sample = 16
    channels = 1
    datasize = samples * channels * bits_per_sample // 8
    o = bytes("RIFF", 'ascii')  # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4, 'little')  # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE", 'ascii')  # (4byte) File type
    o += bytes("fmt ", 'ascii')  # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, 'little')  # (4byte) Length of above format data
    o += (1).to_bytes(2, 'little')  # (2byte) Format type (1 - PCM)
    o += channels.to_bytes(2, 'little')  # (2byte)
    o += sample_rate.to_bytes(4, 'little')  # (4byte)
    o += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')  # (4byte)
    o += (channels * bits_per_sample // 8).to_bytes(2, 'little')  # (2byte)
    o += bits_per_sample.to_bytes(2, 'little')  # (2byte)
    o += bytes("data", 'ascii')  # (4byte) Data Chunk Marker
    o += datasize.to_bytes(4, 'little')  # (4byte) Data size in bytes
    return o


def load_audio(byte_array: bytes) -> ndarray:
    return np.frombuffer(byte_array, np.int16).flatten().astype(np.float32) / 32768.0


# returns now UTC timestamp since epoch in millis
def now() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


_remote_client: httpx.Client | None = None


def _get_remote_client() -> httpx.Client:
    global _remote_client
    if _remote_client is None:
        headers = {}
        if whisper_remote_api_key:
            headers['Authorization'] = f'Bearer {whisper_remote_api_key}'
        _remote_client = httpx.Client(
            base_url=whisper_remote_url,
            headers=headers,
            timeout=whisper_remote_timeout,
        )
    return _remote_client


def _transcribe_local(audio_bytes: bytes, lang: str, previous_tokens) -> WhisperResult:
    audio = load_audio(audio_bytes)
    iterator, _ = cfg.model.transcribe(
        audio,
        language=lang,
        task='transcribe',
        word_timestamps=True,
        beam_size=whisper_beam_size,
        initial_prompt=previous_tokens,
        condition_on_previous_text=False,
    )
    res = list(iterator)
    ts_obj = WhisperResult.from_faster_whisper(res)
    log.debug(f'Transcription results:\n{ts_obj}\n{res}')
    return ts_obj


def _transcribe_remote(audio_bytes: bytes, lang: str, previous_tokens) -> WhisperResult:
    duration_s = convert_bytes_to_seconds(audio_bytes)
    wav_bytes = get_wav_header([audio_bytes], chunk_duration_s=duration_s) + audio_bytes

    data = {
        'model': whisper_remote_model,
        'language': lang,
        'response_format': 'verbose_json',
        'timestamp_granularities[]': 'word',
        'temperature': '0',
    }
    if isinstance(previous_tokens, str) and previous_tokens.strip():
        data['prompt'] = previous_tokens.strip()

    files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}

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

    ts_obj = WhisperResult.from_verbose_json(payload)
    log.debug(f'Remote transcription results:\n{ts_obj}\n{payload}')
    return ts_obj


def transcribe(buffer_list: List[bytes], lang: str = 'en', previous_tokens=None) -> WhisperResult:
    if previous_tokens is None:
        previous_tokens = [] if whisper_backend == 'local' else ''
    audio_bytes = b''.join(buffer_list)
    if whisper_backend == 'remote':
        return _transcribe_remote(audio_bytes, lang, previous_tokens)
    return _transcribe_local(audio_bytes, lang, previous_tokens)


def get_lang(lang: str, short=True) -> str:
    if len(lang) == 2 and short:
        return lang.lower().strip()
    if '-' in lang and short:
        return lang.split('-')[0].strip()
    if not short and '-' in lang:
        split_key = lang.split('-')[0]
        return LANGUAGES.get(split_key, 'english').lower().strip()
    return lang.lower().strip()


class Uuid7:
    def __init__(self):
        self.last_v7_timestamp = None

    def get(self, time_arg_millis: int = None) -> UUID:
        nanoseconds = time.time_ns()
        timestamp_ms = nanoseconds // 10**6

        if time_arg_millis is not None:
            timestamp_ms = time_arg_millis

        if self.last_v7_timestamp is not None and timestamp_ms <= self.last_v7_timestamp:
            timestamp_ms = self.last_v7_timestamp + 1
        self.last_v7_timestamp = timestamp_ms
        uuid_int = (timestamp_ms & 0xFFFFFFFFFFFF) << 80
        uuid_int |= secrets.randbits(76)
        return UUID(int=uuid_int, version=7)


def get_jwt(ws_headers, ws_url_param=None) -> str:
    auth_header = ws_headers.get('authorization', None)
    if auth_header is not None:
        return auth_header.split(' ')[-1]
    return ws_url_param if ws_url_param is not None else ''
