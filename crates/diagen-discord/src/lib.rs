pub const PCM_SAMPLE_RATE: u32 = 48_000;
pub const PCM_CHANNELS: u8 = 2;

#[derive(Debug, thiserror::Error)]
pub enum DiscordError {
    #[error("not yet implemented — backend selected by spike-discord")]
    NotImplemented,
}

pub fn upmix_mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(mono.len() * 2);
    for &s in mono {
        out.push(s);
        out.push(s);
    }
    out
}

pub fn downmix_stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    debug_assert_eq!(stereo.len() % 2, 0);
    let mut out = Vec::with_capacity(stereo.len() / 2);
    for chunk in stereo.chunks_exact(2) {
        out.push((chunk[0] + chunk[1]) * 0.5);
    }
    out
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upmix_doubles_length() {
        let mono = vec![0.1f32, 0.2, 0.3];
        let stereo = upmix_mono_to_stereo(&mono);
        assert_eq!(stereo.len(), 6);
        assert_eq!(stereo[0], 0.1);
        assert_eq!(stereo[1], 0.1);
        assert_eq!(stereo[2], 0.2);
        assert_eq!(stereo[4], 0.3);
    }

    #[test]
    fn downmix_halves_length() {
        let stereo = vec![0.5f32, 0.5, 1.0, -1.0];
        let mono = downmix_stereo_to_mono(&stereo);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!(mono[1].abs() < 1e-6);
    }

    #[test]
    fn rms_known_values() {
        let silent = vec![0.001f32; 960];
        let loud = vec![0.5f32; 960];
        assert!(rms(&silent) < 0.01);
        assert!(rms(&loud) > 0.01);
    }
}
