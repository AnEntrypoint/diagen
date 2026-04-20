import json
import sys
import os
import base64
import numpy as np

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

print('[pocket-tts] Loading model...', file=sys.stderr, flush=True)
from pocket_tts import TTSModel

model = TTSModel.load_model()
sample_rate = int(getattr(model, 'sample_rate', 24000))
print(f'[pocket-tts] Model ready sr={sample_rate}', file=sys.stderr, flush=True)

_state_cache = {}

def get_voice_state(ref_audio_path):
    key = ref_audio_path or '__default__'
    st = _state_cache.get(key)
    if st is not None:
        return st
    src = ref_audio_path if ref_audio_path else 'alba'
    st = model.get_state_for_audio_prompt(src)
    _state_cache[key] = st
    return st

def to_int16_bytes(t):
    arr = t.detach().cpu().numpy().astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()

def handle_generate(req_id, text, ref_audio_path, streaming):
    state = get_voice_state(ref_audio_path)
    if streaming:
        for chunk in model.generate_audio_stream(state, text):
            sys.stdout.write(json.dumps({
                'id': req_id,
                'chunk': True,
                'audio_b64': base64.b64encode(to_int16_bytes(chunk)).decode('utf8'),
                'sample_rate': sample_rate,
            }) + '\n')
            sys.stdout.flush()
        sys.stdout.write(json.dumps({'id': req_id, 'done': True, 'sample_rate': sample_rate}) + '\n')
        sys.stdout.flush()
    else:
        audio = model.generate_audio(state, text)
        sys.stdout.write(json.dumps({
            'id': req_id,
            'success': True,
            'audio_b64': base64.b64encode(to_int16_bytes(audio)).decode('utf8'),
            'sample_rate': sample_rate,
        }) + '\n')
        sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    req_id = None
    try:
        req = json.loads(line)
        req_id = req.get('id')
        text = req['text']
        ref_audio_path = req.get('ref_audio_path') or None
        streaming = bool(req.get('streaming', False))
        handle_generate(req_id, text, ref_audio_path, streaming)
    except Exception as e:
        sys.stdout.write(json.dumps({'id': req_id, 'success': False, 'error': str(e)}) + '\n')
        sys.stdout.flush()
