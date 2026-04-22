use tokio_util::sync::CancellationToken;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("not yet implemented — backend selected by spike-llm")]
    NotImplemented,
}

#[derive(Default, Clone)]
pub struct GenOpts {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub grammar: Option<String>,
}

pub async fn generate(_prompt: &str, _system: Option<&str>, _opts: GenOpts, _abort: CancellationToken) -> Result<String, LlmError> {
    Err(LlmError::NotImplemented)
}

pub async fn is_available() -> bool { false }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_opts_have_no_grammar() {
        let o = GenOpts::default();
        assert!(o.grammar.is_none());
        assert!(o.max_tokens.is_none());
    }
}
