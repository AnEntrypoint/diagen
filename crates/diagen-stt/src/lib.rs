pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("not yet implemented — backend selected by spike-stt")]
    NotImplemented,
}

pub fn is_sentinel(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() { return true; }
    let bytes = t.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i] == b' ' { i += 1; }
        if i >= bytes.len() { break; }
        let open = bytes[i];
        if open != b'[' && open != b'*' {
            return !t.chars().any(|c| c.is_ascii_alphanumeric());
        }
        let close = if open == b'[' { b']' } else { b'*' };
        match t[i + 1..].find(close as char) {
            Some(rel) => i += rel + 2,
            None => return !t.chars().any(|c| c.is_ascii_alphanumeric()),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_sentinels() {
        assert!(is_sentinel("[BLANK_AUDIO]"));
        assert!(is_sentinel("*music*"));
        assert!(is_sentinel("   "));
        assert!(is_sentinel(""));
        assert!(!is_sentinel("hello world"));
    }
}
