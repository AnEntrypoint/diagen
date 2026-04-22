use tokio_util::sync::CancellationToken;

pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("not yet implemented — backend selected by spike-tts")]
    NotImplemented,
}

pub struct SynthesisRequest<'a> {
    pub text: &'a str,
    pub ref_audio_path: Option<&'a str>,
    pub ref_text: Option<&'a str>,
}

pub async fn synthesize(_req: SynthesisRequest<'_>, _abort: CancellationToken) -> Result<Vec<f32>, TtsError> {
    Err(TtsError::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_is_24khz() {
        assert_eq!(SAMPLE_RATE, 24_000);
    }
}
