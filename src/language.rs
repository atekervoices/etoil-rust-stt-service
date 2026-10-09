use serde::{Deserialize, Serialize};

/// Audio streaming constants for Nemotron 3.5 cache-aware ASR
pub const SAMPLE_RATE: usize = 16000;
pub const CHUNK_DURATION_MS: usize = 560;
/// 560ms chunks at 16kHz = 8960 audio samples
pub const CHUNK_SIZE: usize = 8960;

/// Languages supported by Crane Nemotron ASR service.
/// Explicitly excludes Shona ("sn") as requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SupportedLanguage {
    #[serde(rename = "lg")]
    Luganda,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "auto")]
    Auto,
}

impl SupportedLanguage {
    /// Parse and validate user-supplied language code.
    /// Rejects Shona ("sn") and unsupported languages.
    pub fn parse(lang: &str) -> Result<Self, String> {
        let normalized = lang.trim().to_lowercase();
        match normalized.as_str() {
            "lg" | "lug" | "luganda" => Ok(SupportedLanguage::Luganda),
            "en" | "eng" | "english" | "en-us" | "en-gb" | "en_us" | "en_gb" => {
                Ok(SupportedLanguage::English)
            }
            "auto" | "" | "default" => Ok(SupportedLanguage::Auto),
            "sn" | "sna" | "shona" => Err(
                "Language 'shona' (sn) is explicitly disabled in this service configuration."
                    .to_string(),
            ),
            other => Err(format!(
                "Unsupported language '{other}'. This service supports Luganda ('lg', 'luganda'), English ('en', 'english'), or 'auto'. Shona ('sn') is excluded."
            )),
        }
    }

    /// Target language code configured in Nemotron (parakeet-rs).
    ///
    /// Per Crane AI Labs model card for crane-nemo-asr-1.1:
    /// "Best results use the language-ID prompt set to auto-detect;
    ///  Luganda and Shona do not have a dedicated prompt slot, so auto-detect is recommended for them."
    pub fn to_nemotron_lang(self) -> &'static str {
        match self {
            SupportedLanguage::Luganda => "auto",
            SupportedLanguage::English => "en-US",
            SupportedLanguage::Auto => "auto",
        }
    }

    /// Primary standard language code
    pub fn code(self) -> &'static str {
        match self {
            SupportedLanguage::Luganda => "lg",
            SupportedLanguage::English => "en",
            SupportedLanguage::Auto => "auto",
        }
    }

    /// Human readable name
    pub fn name(self) -> &'static str {
        match self {
            SupportedLanguage::Luganda => "Luganda",
            SupportedLanguage::English => "English",
            SupportedLanguage::Auto => "Auto-Detect",
        }
    }
}

impl std::fmt::Display for SupportedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}
