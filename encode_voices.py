#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

import sphn
import ptts


def extract_kv_caches(state):
    num_layers = 6
    tensors = {}
    transformer_state = state._tts_state.flow_lm_state.transformer_state
    for i, layer_state in enumerate(transformer_state.layer_states[:num_layers]):
        mha = layer_state
        k = torch.from_numpy(np.array(mha.k_cache.to_cpu().numpy())).float()
        v = torch.from_numpy(np.array(mha.v_cache.to_cpu().numpy())).float()
        current_end = int(mha.current_end)
        cache = torch.stack([k, v], dim=0)
        if cache.dim() == 4:
            cache = cache.unsqueeze(1)
        tensors[f"transformer.layers.{i}.self_attn/cache"] = cache
        tensors[f"transformer.layers.{i}.self_attn/current_end"] = torch.tensor(float(current_end))
    return tensors


def encode_voice(model, wav_path):
    audio, _ = sphn.read(str(wav_path), sample_rate=24000)
    audio = audio[0, :24000 * 10]
    state = model.get_state_for_audio(audio)
    return extract_kv_caches(state)


def main():
    voices_dir = Path("gh-pages-src/demo/voices")
    wav_files = sorted(voices_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in gh-pages-src/demo/voices/")
        return

    print("Loading model...")
    model = ptts.load_model()
    print(f"Model loaded, voices={model.voices()}")

    for wav_path in wav_files:
        name = wav_path.stem
        out = voices_dir / f"{name}.safetensors"
        print(f"Encoding {name}...")
        try:
            tensors = encode_voice(model, wav_path)
            save_file(tensors, str(out))
            size_kb = out.stat().st_size / 1024
            print(f"  saved {out.name} ({size_kb:.1f} KB)")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("Done!")


if __name__ == "__main__":
    main()
