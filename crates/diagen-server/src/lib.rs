pub const DEFAULT_PORT: u16 = 8080;

pub fn debug_path(subsystem: &str) -> String {
    format!("/debug/{subsystem}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_path_format() {
        assert_eq!(debug_path("speak-gate"), "/debug/speak-gate");
        assert_eq!(debug_path("discord"), "/debug/discord");
    }
}
