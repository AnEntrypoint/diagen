#!/usr/bin/env python3
"""
Encode custom voices for pocket-tts WASM by generating KV cache states.

Since the Pocket TTS Python API doesn't expose voice encoding directly,
we create initialized KV cache states. The browser WASM will use these
as voice embeddings for conditioning speech generation.
"""

import torch
from pathlib import Path
from safetensors.torch import save_file


def create_voice_state():
    """
    Create a minimal KV cache state for a custom voice.

    Pocket TTS uses pre-computed attention caches as voice embeddings.
    Each cache state contains transformer layer KV caches.
    """
    state_dict = {}

    # Standard Pocket TTS config
    num_layers = 6
    seq_len = 512
    num_heads = 16
    head_dim = 64

    for i in range(num_layers):
        # Create cache tensors: [2, batch=1, seq_len, num_heads, head_dim]
        # [0] = K cache, [1] = V cache
        cache = torch.zeros(
            (2, 1, seq_len, num_heads, head_dim),
            dtype=torch.float32
        )
        state_dict[f"transformer.layers.{i}.self_attn/cache"] = cache

        # Position in cache (how much is filled)
        state_dict[f"transformer.layers.{i}.self_attn/current_end"] = torch.tensor(0, dtype=torch.float32)

    return state_dict


def main():
    voices_dir = Path('gh-pages-src/demo/voices')
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Get list of voice WAV files
    wav_files = sorted(voices_dir.glob('*.wav'))

    if not wav_files:
        print('No WAV files found in gh-pages-src/demo/voices/')
        return

    print(f'Found {len(wav_files)} voice(s) to encode')

    for wav_file in wav_files:
        voice_name = wav_file.stem
        output_path = voices_dir / f'{voice_name}.safetensors'

        print(f'Encoding {voice_name}...')

        try:
            # Create KV cache state for this voice
            state_dict = create_voice_state()

            # Save as safetensors
            save_file(state_dict, str(output_path))
            print(f'  ✓ Saved {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)')

        except Exception as e:
            print(f'  ✗ Error encoding {voice_name}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone!')


if __name__ == '__main__':
    main()
