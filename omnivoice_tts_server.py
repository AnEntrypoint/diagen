import json
import sys
import os
import base64
import tempfile
import torch
import torchaudio
from omnivoice.models.omnivoice import OmniVoice

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[omnivoice] Loading model on {device}...', file=sys.stderr, flush=True)

model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map=device, dtype=torch.float16)
print('[omnivoice] Model ready', file=sys.stderr, flush=True)

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        req = json.loads(line)
        text = req['text']
        ref_audio_b64 = req.get('ref_audio_b64')
        ref_text = req.get('ref_text')

        ref_audio_path = None
        if ref_audio_b64 and ref_text:
            audio_bytes = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_bytes)
                ref_audio_path = f.name

        audios = model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            num_step=32,
            guidance_scale=2.0,
            speed=1.0,
            denoise=True,
            postprocess_output=True
        )

        audio_output = audios[0].cpu().numpy()
        audio_int16 = (audio_output * 32767).astype('int16')
        audio_bytes = audio_int16.tobytes()

        response = {
            'success': True,
            'audio_b64': base64.b64encode(audio_bytes).decode('utf8'),
            'sample_rate': model.sampling_rate
        }
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

        if ref_audio_path and os.path.exists(ref_audio_path):
            try:
                os.unlink(ref_audio_path)
            except:
                pass

    except Exception as e:
        response = {'success': False, 'error': str(e)}
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()
