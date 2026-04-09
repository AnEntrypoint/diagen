#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
import sphn
from pocket_tts.models.tts_model import TTSModel, RECOMMENDED_CONFIG, init_states, get_flow_lm_state_dict
from pocket_tts import export_model_state


def load_model():
    model = TTSModel(RECOMMENDED_CONFIG)
    model.load()
    return model


def encode_voice(model, wav_path):
    audio, _ = sphn.read(str(wav_path), sample_rate=24000)
    audio = audio[0, :24000 * 10]
    states = init_states(model, audio=audio)
    return states_to_tensors(states)


def states_to_tensors(states):
    state_dict = get_flow_lm_state_dict(states)
    tensors = {}
    num_layers = 6
    for i in range(num_layers):
        k_key = f"transformer.layers.{i}.self_attn.k_cache"
        v_key = f"transformer.layers.{i}.self_attn.v_cache"
        end_key = f"transformer.layers.{i}.self_attn.current_end"
        k = state_dict.get(k_key)
        v = state_dict.get(v_key)
        end = state_dict.get(end_key, torch.tensor(0.0))
        if k is None or v is None:
            print(f"  WARNING: missing keys for layer {i}, available: {list(state_dict.keys())[:10]}")
            raise KeyError(f"Missing k/v cache for layer {i}")
        cache = torch.stack([k, v], dim=0)
        if cache.dim() == 4:
            cache = cache.unsqueeze(1)
        tensors[f"transformer.layers.{i}.self_attn/cache"] = cache.float()
        tensors[f"transformer.layers.{i}.self_attn/current_end"] = end.float().reshape(())
    return tensors


def main():
    voices_dir = Path("gh-pages-src/demo/voices")
    wav_files = sorted(voices_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found")
        return

    print("Loading model...")
    model = load_model()
    print(f"Model loaded: {type(model)}")

    for wav_path in wav_files:
        name = wav_path.stem
        out = voices_dir / f"{name}.safetensors"
        print(f"Encoding {name}...")
        try:
            tensors = encode_voice(model, wav_path)
            save_file(tensors, str(out))
            print(f"  saved {out.name} ({out.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("Done!")


if __name__ == "__main__":
    main()
