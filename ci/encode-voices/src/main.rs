use ptts::transformer::LayerAttentionState;
use ptts::tts_model::{TTSConfig, TTSModel};
use xn::nn::VB;
use xn::{CPU, TypedTensor};
use std::collections::HashMap;
use std::path::PathBuf;

struct NoopTokenizer;
impl ptts::Tokenizer for NoopTokenizer {
    fn encode(&self, _text: &str) -> Vec<u32> { vec![] }
    fn decode(&self, _tokens: &[u32]) -> String { String::new() }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t") || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj") || name.contains("learnt_padding") {
        return None;
    }
    let mut n = name.to_string();
    n = n.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    n = n.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    n = n.replace("flow_lm.backbone.", "flow_lm.transformer.");
    n = n.replace("flow_lm.flow.", "flow_lm.flow_net.");
    n = n.replace("mimi.model.", "mimi.");
    Some(n)
}

fn read_wav_mono_24k(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let num_channels = spec.channels as usize;
    let mono: Vec<f32> = samples.chunks(num_channels)
        .map(|c| c.iter().sum::<f32>() / num_channels as f32)
        .collect();
    let target_len = 24000 * 10;
    if spec.sample_rate == 24000 {
        mono.into_iter().take(target_len).collect()
    } else {
        let ratio = spec.sample_rate as f64 / 24000.0;
        let out_len = ((mono.len() as f64 / ratio) as usize).min(target_len);
        (0..out_len).map(|i| {
            let idx = i as f64 * ratio;
            let lo = idx.floor() as usize;
            let hi = (lo + 1).min(mono.len() - 1);
            let frac = (idx - idx.floor()) as f32;
            mono[lo] * (1.0 - frac) + mono[hi] * frac
        }).collect()
    }
}

fn encode_voice(
    model: &TTSModel<f32, xn::CpuDevice>,
    wav_path: &std::path::Path,
    out_path: &std::path::Path,
) -> xn::Result<()> {
    let pcm = read_wav_mono_24k(wav_path);
    println!("  {} samples", pcm.len());
    let audio = xn::Tensor::from_vec(pcm, (1usize, 1usize, ()), &CPU)?;
    let voice_emb = model.encode_audio(&audio)?;
    let mut state = model.init_flow_lm_state(1, 512)?;
    model.prompt_audio(&mut state, &voice_emb)?;

    let layers = &state.flow_lm_state.transformer_state.layer_states;
    let mut tensors: HashMap<String, TypedTensor<xn::CpuDevice>> = HashMap::new();
    for (i, layer) in layers.iter().enumerate() {
        if let LayerAttentionState::FlowLm(mha) = layer {
            let end = mha.current_end;
            if end == 0 {
                return Err(xn::Error::msg(format!("layer {i} current_end=0 after prompt_audio")));
            }
            let k = mha.k_cache.contiguous()?;
            let v = mha.v_cache.contiguous()?;
            let (b, full_seq, h, d) = k.dims4()?;
            let k5 = k.reshape((1usize, b, full_seq, h, d))?;
            let v5 = v.reshape((1usize, b, full_seq, h, d))?;
            let cache = xn::Tensor::cat(&[&k5, &v5], 0)?;
            tensors.insert(
                format!("transformer.layers.{i}.self_attn/cache"),
                TypedTensor::F32(cache),
            );
            let end_tensor = xn::Tensor::from_vec(vec![end as f32], (1usize,), &CPU)?;
            tensors.insert(
                format!("transformer.layers.{i}.self_attn/current_end"),
                TypedTensor::F32(end_tensor),
            );
        }
    }
    xn::safetensors::save(&tensors, out_path)?;
    println!("  saved {:?}", out_path);
    Ok(())
}

fn main() {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    println!("Downloading model...");
    let api = Api::new().expect("hf api");
    let repo = api.repo(Repo::new("kyutai/pocket-tts-without-voice-cloning".to_string(), RepoType::Model));
    let model_path = repo.get("tts_b6369a24.safetensors").expect("download model");
    println!("Model at {:?}", model_path);

    let cfg = TTSConfig::v202601(0.7);
    let vb = VB::load_with_key_map(&[&model_path], CPU, remap_key)
        .expect("load weights")
        .root();
    let model = TTSModel::<f32, xn::CpuDevice>::load(&vb, Box::new(NoopTokenizer), &cfg)
        .expect("build model");
    println!("Model loaded");

    let voices_dir = PathBuf::from("gh-pages-src/demo/voices");
    let mut entries: Vec<_> = std::fs::read_dir(&voices_dir)
        .expect("voices dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        println!("No WAV files found");
        return;
    }

    for entry in entries {
        let wav_path = entry.path();
        let name = wav_path.file_stem().unwrap().to_str().unwrap().to_string();
        let out_path = voices_dir.join(format!("{name}.safetensors"));
        println!("Encoding {}...", name);
        match encode_voice(&model, &wav_path, &out_path) {
            Ok(()) => println!("  OK"),
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }
    println!("Done!");
}
