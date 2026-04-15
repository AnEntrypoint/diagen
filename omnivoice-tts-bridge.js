import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const OMNIVOICE_REPO = 'C:\\dev\\omnivoice'

let ttsProcess = null
let processReady = false
let readyTimeout = null

function startTtsProcess() {
  if (ttsProcess) return Promise.resolve()

  return new Promise((resolve, reject) => {
    const script = `
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
    sys.stdout.write(json.dumps(response) + '\\n')
    sys.stdout.flush()

    if ref_audio_path and os.path.exists(ref_audio_path):
      try:
        os.unlink(ref_audio_path)
      except:
        pass

  except Exception as e:
    response = {'success': False, 'error': str(e)}
    sys.stdout.write(json.dumps(response) + '\\n')
    sys.stdout.flush()
`

    // Use uv run to execute within the OmniVoice virtual environment
    ttsProcess = spawn('uv', ['run', 'python', '-c', script], {
      cwd: OMNIVOICE_REPO,
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 180000
    })

    let stderrOutput = ''
    readyTimeout = setTimeout(() => {
      readyTimeout = null
      if (ttsProcess) ttsProcess.kill()
      reject(new Error('OmniVoice server startup timeout (600s - model download may be in progress)'))
    }, 600000)

    ttsProcess.stderr.on('data', (chunk) => {
      stderrOutput += chunk.toString()
      if (stderrOutput.includes('[omnivoice] Model ready')) {
        if (readyTimeout) {
          clearTimeout(readyTimeout)
          readyTimeout = null
        }
        processReady = true
        resolve()
      }
    })

    ttsProcess.on('error', (err) => {
      processReady = false
      if (readyTimeout) {
        clearTimeout(readyTimeout)
        readyTimeout = null
      }
      reject(new Error(`TTS process spawn failed: ${err.message}`))
    })

    ttsProcess.on('exit', (code) => {
      processReady = false
      ttsProcess = null
      if (readyTimeout) {
        clearTimeout(readyTimeout)
        readyTimeout = null
      }
    })
  })
}

export async function synthesize(text, refAudioPath, refText) {
  if (!text) throw new Error('text required for synthesis');

  await startTtsProcess();

  return new Promise((resolve, reject) => {
    let responseData = '';
    const timeout = setTimeout(() => {
      responseListener?.();
      reject(new Error('TTS synthesis timeout (300s - model may be downloading)'));
    }, 300000);

    const responseListener = ttsProcess.stdout.once('data', (chunk) => {
      clearTimeout(timeout);
      responseData += chunk.toString();

      try {
        const lines = responseData.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        const response = JSON.parse(lastLine);

        if (!response.success) {
          throw new Error(`TTS failed: ${response.error}`);
        }

        const audioBuffer = Buffer.from(response.audio_b64, 'base64');
        const int16Array = new Int16Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.length / 2);
        const float32Array = new Float32Array(int16Array.length);

        for (let i = 0; i < int16Array.length; i++) {
          float32Array[i] = int16Array[i] / 32768;
        }

        resolve(float32Array);
      } catch (err) {
        reject(new Error(`TTS response parse failed: ${err.message}`));
      }
    });

    try {
      const refAudioB64 = refAudioPath ? require('fs').readFileSync(refAudioPath).toString('base64') : null;

      const request = {
        text,
        ref_audio_b64: refAudioB64,
        ref_text: refText
      };

      ttsProcess.stdin.write(JSON.stringify(request) + '\n');
    } catch (err) {
      clearTimeout(timeout);
      reject(new Error(`TTS request send failed: ${err.message}`));
    }
  });
}

export function shutdown() {
  if (ttsProcess) {
    ttsProcess.kill();
    ttsProcess = null;
    processReady = false;
  }
}

export default { synthesize, shutdown };
