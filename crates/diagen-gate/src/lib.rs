use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum State {
    Listening,
    Waiting,
    Gating,
    Answering,
    Speaking,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Listening => "LISTENING",
            State::Waiting => "WAITING",
            State::Gating => "GATING",
            State::Answering => "ANSWERING",
            State::Speaking => "SPEAKING",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_names_match_node_constants() {
        assert_eq!(State::Listening.name(), "LISTENING");
        assert_eq!(State::Waiting.name(), "WAITING");
        assert_eq!(State::Gating.name(), "GATING");
        assert_eq!(State::Answering.name(), "ANSWERING");
        assert_eq!(State::Speaking.name(), "SPEAKING");
    }
}
