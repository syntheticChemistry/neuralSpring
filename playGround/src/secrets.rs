// SPDX-License-Identifier: AGPL-3.0-or-later

//! Load API keys from `ecoPrimals/testing-secrets/api-keys.toml`.
//!
//! The file has a mix of loose key-value pairs (top section) and standard
//! TOML sections. This module handles both formats.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// All API keys loaded from testing-secrets.
#[derive(Debug, Default)]
pub struct Secrets {
    /// Hugging Face API token for Hub downloads (when present in the file).
    pub huggingface_token: Option<String>,
    /// Anthropic API key from `[ai_providers]` or similar sections.
    pub anthropic_api_key: Option<String>,
    /// `OpenAI` API key from `[ai_providers]` or similar sections.
    pub openai_api_key: Option<String>,
    /// Cohere API key parsed from loose key lines.
    pub cohere_api_key: Option<String>,
    /// Together AI API key parsed from loose key lines.
    pub together_api_key: Option<String>,
    sections: HashMap<String, HashMap<String, String>>,
}

impl Secrets {
    /// Resolve the default secrets path relative to the workspace.
    #[must_use]
    pub fn default_path() -> PathBuf {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| Path::new("."));
        workspace
            .parent()
            .unwrap_or(workspace)
            .join("testing-secrets")
            .join("api-keys.toml")
    }

    /// Load secrets from a file path.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading secrets from {}", path.display()))?;
        Ok(Self::parse(&content))
    }

    /// Load from the default path.
    pub fn load_default() -> Result<Self> {
        Self::load(&Self::default_path())
    }

    fn parse(content: &str) -> Self {
        let mut secrets = Self::default();
        let mut loose_lines: Vec<String> = Vec::new();

        let mut toml_start = None;
        for (i, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && !trimmed.starts_with('#') {
                toml_start = Some(i);
                break;
            }
            loose_lines.push(line.to_string());
        }

        secrets.parse_loose_keys(&loose_lines);

        if let Some(start) = toml_start {
            let toml_content: String = content.lines().skip(start).collect::<Vec<_>>().join("\n");
            if let Ok(table) = toml_content.parse::<toml::Table>() {
                secrets.ingest_toml(&table);
            }
        }

        secrets
    }

    fn parse_loose_keys(&mut self, lines: &[String]) {
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            // Format: "key : value" on one line
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_lowercase();
                let value = value.trim().to_string();
                self.assign_loose_key(&key, &value);
                i += 1;
                continue;
            }

            // Format: key label on one line, value on next
            let key = line.to_lowercase();
            if i + 1 < lines.len() {
                let next = lines[i + 1].trim().to_string();
                if !next.is_empty() && !next.starts_with('#') && !next.starts_with('[') {
                    self.assign_loose_key(&key, &next);
                    i += 2;
                    continue;
                }
            }

            i += 1;
        }
    }

    fn assign_loose_key(&mut self, key: &str, value: &str) {
        if key.contains("hugging") || value.starts_with("hf_") {
            self.huggingface_token = Some(value.to_string());
        } else if key.contains("cohere") {
            self.cohere_api_key = Some(value.to_string());
        } else if key.contains("together") {
            self.together_api_key = Some(value.to_string());
        }
    }

    fn ingest_toml(&mut self, table: &toml::Table) {
        for (section, value) in table {
            if let Some(inner) = value.as_table() {
                let mut section_map = HashMap::new();
                for (k, v) in inner {
                    if let Some(s) = v.as_str() {
                        section_map.insert(k.clone(), s.to_string());
                    }
                }
                self.sections.insert(section.clone(), section_map);
            }
        }

        if let Some(ai) = self.sections.get("ai_providers") {
            if let Some(key) = ai.get("anthropic_api_key") {
                self.anthropic_api_key = Some(key.clone());
            }
            if let Some(key) = ai.get("openai_api_key") {
                self.openai_api_key = Some(key.clone());
            }
        }
    }

    /// Get any key from a specific TOML section.
    #[must_use]
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections
            .get(section)
            .and_then(|s| s.get(key))
            .map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mixed_format() {
        let content = r#"
# comment
hugging face
hf_TestToken123

[ai_providers]
anthropic_api_key = "sk-ant-test"
openai_api_key = "sk-proj-test"
"#;
        let secrets = Secrets::parse(content);
        assert_eq!(
            secrets.huggingface_token.as_deref(),
            Some("hf_TestToken123")
        );
        assert_eq!(secrets.anthropic_api_key.as_deref(), Some("sk-ant-test"));
        assert_eq!(secrets.openai_api_key.as_deref(), Some("sk-proj-test"));
    }

    #[test]
    fn parse_colon_format() {
        let content = "cloudflareAPI : abc123\n";
        let secrets = Secrets::parse(content);
        assert!(secrets.huggingface_token.is_none());
    }
}
