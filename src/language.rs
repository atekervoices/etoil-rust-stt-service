use serde::{Deserialize, Serialize};

/// Audio streaming constants for Nemotron 3.5 cache-aware ASR
pub const SAMPLE_RATE: usize = 16000;
pub const CHUNK_DURATION_MS: usize = 560;
/// 560ms chunks at 16kHz = 8960 audio samples (right-context 6, matching trained lookahead)
pub const CHUNK_SIZE: usize = 8960;

/// Languages supported by NVIDIA Nemotron 3.5 Streaming Multilingual ASR
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SupportedLanguage {
    #[serde(rename = "en")]
    English,
    #[serde(rename = "en-gb")]
    EnglishGB,
    #[serde(rename = "es")]
    Spanish,
    #[serde(rename = "fr")]
    French,
    #[serde(rename = "de")]
    German,
    #[serde(rename = "it")]
    Italian,
    #[serde(rename = "pt")]
    Portuguese,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "ja")]
    Japanese,
    #[serde(rename = "ko")]
    Korean,
    #[serde(rename = "hi")]
    Hindi,
    #[serde(rename = "ar")]
    Arabic,
    #[serde(rename = "tr")]
    Turkish,
    #[serde(rename = "vi")]
    Vietnamese,
    #[serde(rename = "uk")]
    Ukrainian,
    #[serde(rename = "nl")]
    Dutch,
    #[serde(rename = "pl")]
    Polish,
    #[serde(rename = "sv")]
    Swedish,
    #[serde(rename = "sw")]
    Swahili,
    #[serde(rename = "lug")]
    Luganda,
    #[serde(rename = "auto")]
    Auto,
    Custom(String, String),
}

impl SupportedLanguage {
    /// Parse and validate user-supplied language code.
    pub fn parse(lang: &str) -> Result<Self, String> {
        let normalized = lang.trim().to_lowercase();
        match normalized.as_str() {
            "en" | "eng" | "english" | "en-us" | "en_us" | "us" | "american" => {
                Ok(SupportedLanguage::English)
            }
            "en-gb" | "eng-gb" | "en_gb" | "british" | "commonwealth" => {
                Ok(SupportedLanguage::EnglishGB)
            }
            "es" | "spa" | "spanish" | "es-es" | "es_es" => Ok(SupportedLanguage::Spanish),
            "fr" | "fra" | "fre" | "french" | "fr-fr" | "fr_fr" => Ok(SupportedLanguage::French),
            "de" | "deu" | "ger" | "german" | "de-de" | "de_de" => Ok(SupportedLanguage::German),
            "it" | "ita" | "italian" | "it-it" | "it_it" => Ok(SupportedLanguage::Italian),
            "pt" | "por" | "portuguese" | "pt-br" | "pt_br" => Ok(SupportedLanguage::Portuguese),
            "ru" | "rus" | "russian" | "ru-ru" | "ru_ru" => Ok(SupportedLanguage::Russian),
            "zh" | "zho" | "chi" | "chinese" | "zh-cn" | "zh_cn" => {
                Ok(SupportedLanguage::Chinese)
            }
            "ja" | "jpn" | "japanese" | "ja-jp" | "ja_jp" => Ok(SupportedLanguage::Japanese),
            "ko" | "kor" | "korean" | "ko-kr" | "ko_kr" => Ok(SupportedLanguage::Korean),
            "hi" | "hin" | "hindi" | "hi-in" | "hi_in" => Ok(SupportedLanguage::Hindi),
            "ar" | "ara" | "arabic" | "ar-ar" | "ar_ar" => Ok(SupportedLanguage::Arabic),
            "tr" | "tur" | "turkish" | "tr-tr" | "tr_tr" => Ok(SupportedLanguage::Turkish),
            "vi" | "vie" | "vietnamese" | "vi-vn" | "vi_vn" => Ok(SupportedLanguage::Vietnamese),
            "uk" | "ukr" | "ukrainian" | "uk-ua" | "uk_ua" => Ok(SupportedLanguage::Ukrainian),
            "nl" | "nld" | "dutch" | "nl-nl" | "nl_nl" => Ok(SupportedLanguage::Dutch),
            "pl" | "pol" | "polish" | "pl-pl" | "pl_pl" => Ok(SupportedLanguage::Polish),
            "sv" | "swe" | "swedish" | "sv-se" | "sv_se" => Ok(SupportedLanguage::Swedish),
            "sw" | "swa" | "swahili" | "sw-ke" | "sw_ke" => Ok(SupportedLanguage::Swahili),
            "lug" | "luganda" | "ganda" | "lg" | "lug-ug" | "lug_ug" => {
                Ok(SupportedLanguage::Luganda)
            }
            "auto" | "" | "default" => Ok(SupportedLanguage::Auto),
            other => Ok(SupportedLanguage::Custom(other.to_string(), other.to_string())),
        }
    }

    /// Target language code configured in Nemotron (parakeet-rs prompt dictionary).
    pub fn to_nemotron_lang(&self) -> &str {
        match self {
            SupportedLanguage::English => "en-US",
            SupportedLanguage::EnglishGB => "en-GB",
            SupportedLanguage::Spanish => "es-ES",
            SupportedLanguage::French => "fr-FR",
            SupportedLanguage::German => "de-DE",
            SupportedLanguage::Italian => "it-IT",
            SupportedLanguage::Portuguese => "pt-BR",
            SupportedLanguage::Russian => "ru-RU",
            SupportedLanguage::Chinese => "zh-CN",
            SupportedLanguage::Japanese => "ja-JP",
            SupportedLanguage::Korean => "ko-KR",
            SupportedLanguage::Hindi => "hi-IN",
            SupportedLanguage::Arabic => "ar-AR",
            SupportedLanguage::Turkish => "tr-TR",
            SupportedLanguage::Vietnamese => "vi-VN",
            SupportedLanguage::Ukrainian => "uk-UA",
            SupportedLanguage::Dutch => "nl-NL",
            SupportedLanguage::Polish => "pl-PL",
            SupportedLanguage::Swedish => "sv-SE",
            SupportedLanguage::Swahili => "sw-KE",
            SupportedLanguage::Luganda => "lug",
            SupportedLanguage::Auto => "auto",
            SupportedLanguage::Custom(_, target) => target.as_str(),
        }
    }

    /// Primary standard language code
    pub fn code(&self) -> &str {
        match self {
            SupportedLanguage::English => "en",
            SupportedLanguage::EnglishGB => "en-gb",
            SupportedLanguage::Spanish => "es",
            SupportedLanguage::French => "fr",
            SupportedLanguage::German => "de",
            SupportedLanguage::Italian => "it",
            SupportedLanguage::Portuguese => "pt",
            SupportedLanguage::Russian => "ru",
            SupportedLanguage::Chinese => "zh",
            SupportedLanguage::Japanese => "ja",
            SupportedLanguage::Korean => "ko",
            SupportedLanguage::Hindi => "hi",
            SupportedLanguage::Arabic => "ar",
            SupportedLanguage::Turkish => "tr",
            SupportedLanguage::Vietnamese => "vi",
            SupportedLanguage::Ukrainian => "uk",
            SupportedLanguage::Dutch => "nl",
            SupportedLanguage::Polish => "pl",
            SupportedLanguage::Swedish => "sv",
            SupportedLanguage::Swahili => "sw",
            SupportedLanguage::Luganda => "lug",
            SupportedLanguage::Auto => "auto",
            SupportedLanguage::Custom(code, _) => code.as_str(),
        }
    }

    /// Human readable name
    pub fn name(&self) -> &str {
        match self {
            SupportedLanguage::English => "English (US)",
            SupportedLanguage::EnglishGB => "English (UK / Commonwealth)",
            SupportedLanguage::Spanish => "Spanish",
            SupportedLanguage::French => "French",
            SupportedLanguage::German => "German",
            SupportedLanguage::Italian => "Italian",
            SupportedLanguage::Portuguese => "Portuguese",
            SupportedLanguage::Russian => "Russian",
            SupportedLanguage::Chinese => "Chinese (Mandarin)",
            SupportedLanguage::Japanese => "Japanese",
            SupportedLanguage::Korean => "Korean",
            SupportedLanguage::Hindi => "Hindi",
            SupportedLanguage::Arabic => "Arabic",
            SupportedLanguage::Turkish => "Turkish",
            SupportedLanguage::Vietnamese => "Vietnamese",
            SupportedLanguage::Ukrainian => "Ukrainian",
            SupportedLanguage::Dutch => "Dutch",
            SupportedLanguage::Polish => "Polish",
            SupportedLanguage::Swedish => "Swedish",
            SupportedLanguage::Swahili => "Swahili",
            SupportedLanguage::Luganda => "Luganda (Ganda)",
            SupportedLanguage::Auto => "Auto-Detect",
            SupportedLanguage::Custom(name, _) => name.as_str(),
        }
    }
}

impl Default for SupportedLanguage {
    fn default() -> Self {
        SupportedLanguage::Auto
    }
}

impl std::fmt::Display for SupportedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}
