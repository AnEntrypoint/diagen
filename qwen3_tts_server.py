import json
import sys
import os
import base64
import time
import numpy as np

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

MODEL_NAME = os.environ.get('QWEN3_TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base')
DEVICE = os.environ.get('QWEN3_TTS_DEVICE', 'cuda')
ATTN = os.environ.get('QWEN3_TTS_ATTN', 'sdpa')
DEFAULT_REF = os.environ.get('QWEN3_TTS_DEFAULT_REF', os.path.join(os.path.dirname(__file__), 'voices', 'cleetus.wav'))

def _read_sidecar(wav_path, suffix):
    if not wav_path:
        return ''
    txt_path = os.path.splitext(wav_path)[0] + suffix
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf8') as f:
            return f.read().strip()
    return ''

DEFAULT_REF_TEXT = os.environ.get('QWEN3_TTS_DEFAULT_REF_TEXT') or _read_sidecar(DEFAULT_REF, '.txt')
LANGUAGE = os.environ.get('QWEN3_TTS_LANGUAGE', 'English')
CHUNK_SIZE = int(os.environ.get('QWEN3_TTS_CHUNK_SIZE', '4'))

print(f'[qwen3-tts] Loading {MODEL_NAME} device={DEVICE} attn={ATTN}...', file=sys.stderr, flush=True)
import torch

# Library prints CUDA-graph status to stdout via print(); rebind stdout->stderr during load + warmup
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    from faster_qwen3_tts import FasterQwen3TTS
    model = FasterQwen3TTS.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16, attn_implementation=ATTN)
finally:
    sys.stdout = _real_stdout
print(f'[qwen3-tts] Model ready device={DEVICE} default_ref={DEFAULT_REF}', file=sys.stderr, flush=True)

def to_int16_bytes(arr):
    a = np.asarray(arr, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype(np.int16).tobytes()

def emit(obj):
    _real_stdout.write(json.dumps(obj) + '\n')
    _real_stdout.flush()

# Permanently redirect stdout->stderr so library prints don't pollute the JSON channel.
sys.stdout = sys.stderr

def resolve_ref(req):
    p = req.get('ref_audio_path')
    if p and os.path.exists(p):
        rt = req.get('ref_text') or _read_sidecar(p, '.txt') or DEFAULT_REF_TEXT
        return p, rt
    if DEFAULT_REF and os.path.exists(DEFAULT_REF):
        return DEFAULT_REF, DEFAULT_REF_TEXT
    raise ValueError('no reference audio available; set QWEN3_TTS_DEFAULT_REF or pass ref_audio_path')

def handle_generate(req_id, text, ref_audio_path, ref_text, streaming):
    t0 = time.time()
    if streaming:
        n = 0
        t_first = None
        total_samples = 0
        sr_observed = None
        for chunk, sr, _timing in model.generate_voice_clone_streaming(
            text=text, language=LANGUAGE, ref_audio=ref_audio_path, ref_text=ref_text, chunk_size=CHUNK_SIZE,
        ):
            if t_first is None:
                t_first = time.time() - t0
            sr_observed = sr
            arr = chunk if isinstance(chunk, np.ndarray) else np.asarray(chunk)
            total_samples += arr.shape[-1] if arr.ndim > 0 else 1
            emit({'id': req_id, 'chunk': True, 'audio_b64': base64.b64encode(to_int16_bytes(arr)).decode('utf8'), 'sample_rate': int(sr)})
            n += 1
        total = time.time() - t0
        audio_sec = (total_samples / sr_observed) if sr_observed else 0
        rt = (audio_sec / total) if total > 0 else 0
        print(f'[qwen3-tts] id={req_id} stream chunks={n} first={t_first or 0:.2f}s total={total:.2f}s audio={audio_sec:.2f}s RT={rt:.2f}x text={text[:40]!r}', file=sys.stderr, flush=True)
        emit({'id': req_id, 'done': True, 'sample_rate': int(sr_observed or 24000)})
    else:
        wavs, sr = model.generate_voice_clone(text=text, language=LANGUAGE, ref_audio=ref_audio_path, ref_text=ref_text)
        arr = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        arr = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
        total = time.time() - t0
        samples = arr.shape[-1] if arr.ndim > 0 else 1
        audio_sec = samples / sr
        rt = audio_sec / total if total > 0 else 0
        print(f'[qwen3-tts] id={req_id} one-shot total={total:.2f}s audio={audio_sec:.2f}s RT={rt:.2f}x text={text[:40]!r}', file=sys.stderr, flush=True)
        emit({'id': req_id, 'success': True, 'audio_b64': base64.b64encode(to_int16_bytes(arr)).decode('utf8'), 'sample_rate': int(sr)})

while True:
    line = sys.stdin.readline()
    if not line:
        break
    req_id = None
    try:
        req = json.loads(line)
        req_id = req.get('id')
        text = req['text']
        ref_audio_path, ref_text = resolve_ref(req)
        streaming = bool(req.get('streaming', False))
        handle_generate(req_id, text, ref_audio_path, ref_text, streaming)
    except Exception as e:
        emit({'id': req_id, 'success': False, 'error': str(e)})
