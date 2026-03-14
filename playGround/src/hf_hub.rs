// SPDX-License-Identifier: AGPL-3.0-or-later

//! `HuggingFace` Hub client for downloading model files.
//!
//! Downloads `config.json` and `*.safetensors` files from HF Hub
//! using the REST API. Supports authentication via HF token.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_DOWNLOAD_BASE: &str = "https://huggingface.co";

/// `HuggingFace` Hub client.
pub struct HfHub {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

/// Metadata about a model from HF Hub.
#[derive(Debug, serde::Deserialize)]
pub struct ModelInfo {
    #[serde(rename = "modelId")]
    pub model_id: String,
    #[serde(default)]
    pub sha: String,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub library_name: Option<String>,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
}

#[derive(Debug, serde::Deserialize)]
pub struct HfSibling {
    #[serde(rename = "rfilename")]
    pub filename: String,
}

impl HfHub {
    /// Create a new Hub client with optional authentication.
    pub fn new(token: Option<&str>, cache_dir: PathBuf) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(t) = token {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {t}"))
                    .context("invalid token")?,
            );
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent("neuralSpring-playGround/0.1.0")
            .build()
            .context("building HTTP client")?;

        Ok(Self { client, cache_dir })
    }

    /// Fetch model metadata from HF Hub.
    pub async fn model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let url = format!("{HF_API_BASE}/{model_id}");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("fetching model info for {model_id}"))?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "HF Hub returned {} for {model_id}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            );
        }

        resp.json::<ModelInfo>().await.context("parsing model info")
    }

    /// List safetensors files in a model.
    pub async fn list_safetensors(&self, model_id: &str) -> Result<Vec<String>> {
        let info = self.model_info(model_id).await?;
        let files: Vec<String> = info
            .siblings
            .into_iter()
            .map(|s| s.filename)
            .filter(|f| f.ends_with(".safetensors"))
            .collect();
        Ok(files)
    }

    /// Download a single file from a model repo.
    pub async fn download_file(&self, model_id: &str, filename: &str) -> Result<PathBuf> {
        let dest_dir = self.cache_dir.join(model_id.replace('/', "--"));
        tokio::fs::create_dir_all(&dest_dir).await?;
        let dest = dest_dir.join(filename);

        if dest.exists() {
            log::info!("Cache hit: {}", dest.display());
            return Ok(dest);
        }

        // Ensure parent dirs exist for nested filenames
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let url = format!("{HF_DOWNLOAD_BASE}/{model_id}/resolve/main/{filename}");
        log::info!("Downloading {url} -> {}", dest.display());

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("downloading {filename} from {model_id}"))?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "HF download returned {} for {model_id}/{filename}",
                resp.status()
            );
        }

        let bytes = resp.bytes().await?;
        tokio::fs::write(&dest, &bytes).await?;

        log::info!("Downloaded {} bytes -> {}", bytes.len(), dest.display());
        Ok(dest)
    }

    /// Download config.json and all safetensors files for a model.
    pub async fn download_model(&self, model_id: &str) -> Result<ModelFiles> {
        let info = self.model_info(model_id).await?;

        let mut config_path = None;
        let mut safetensor_paths = Vec::new();
        let mut tokenizer_path = None;

        let filenames: Vec<String> = info.siblings.iter().map(|s| s.filename.clone()).collect();

        for filename in &filenames {
            let dominated = filename.as_str();
            let should_download = dominated == "config.json"
                || dominated.ends_with(".safetensors")
                || dominated == "tokenizer.json"
                || dominated == "tokenizer_config.json";

            if !should_download {
                continue;
            }

            let path = self.download_file(model_id, dominated).await?;

            if dominated == "config.json" {
                config_path = Some(path);
            } else if dominated.ends_with(".safetensors") {
                safetensor_paths.push(path);
            } else if dominated == "tokenizer.json" {
                tokenizer_path = Some(path);
            }
        }

        Ok(ModelFiles {
            model_id: model_id.to_string(),
            config: config_path,
            safetensors: safetensor_paths,
            tokenizer: tokenizer_path,
        })
    }

    /// Get the cache directory for a model.
    #[must_use]
    pub fn model_cache_dir(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(model_id.replace('/', "--"))
    }
}

/// Paths to downloaded model files.
#[derive(Debug)]
pub struct ModelFiles {
    pub model_id: String,
    pub config: Option<PathBuf>,
    pub safetensors: Vec<PathBuf>,
    pub tokenizer: Option<PathBuf>,
}

impl ModelFiles {
    /// Check if we have the minimum files needed for inference.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.config.is_some() && !self.safetensors.is_empty()
    }
}

/// Default cache directory for downloaded models.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
    workspace.join(".model_cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_dir_is_under_manifest() {
        let dir = default_cache_dir();
        assert!(dir.ends_with(".model_cache"));
        assert!(dir.parent().is_some());
    }

    #[test]
    fn model_cache_dir_normalizes_slashes() {
        let hub = HfHub::new(None, PathBuf::from("/tmp/hf_test")).unwrap();
        let dir = hub.model_cache_dir("openai-community/gpt2");
        assert_eq!(dir, PathBuf::from("/tmp/hf_test/openai-community--gpt2"));
    }

    #[test]
    fn model_cache_dir_handles_simple_id() {
        let hub = HfHub::new(None, PathBuf::from("/tmp/hf_test")).unwrap();
        let dir = hub.model_cache_dir("bert-base-uncased");
        assert_eq!(dir, PathBuf::from("/tmp/hf_test/bert-base-uncased"));
    }

    #[test]
    fn model_files_is_complete() {
        let complete = ModelFiles {
            model_id: "test".into(),
            config: Some(PathBuf::from("config.json")),
            safetensors: vec![PathBuf::from("model.safetensors")],
            tokenizer: None,
        };
        assert!(complete.is_complete());
    }

    #[test]
    fn model_files_incomplete_no_config() {
        let no_config = ModelFiles {
            model_id: "test".into(),
            config: None,
            safetensors: vec![PathBuf::from("model.safetensors")],
            tokenizer: None,
        };
        assert!(!no_config.is_complete());
    }

    #[test]
    fn model_files_incomplete_no_safetensors() {
        let no_weights = ModelFiles {
            model_id: "test".into(),
            config: Some(PathBuf::from("config.json")),
            safetensors: vec![],
            tokenizer: None,
        };
        assert!(!no_weights.is_complete());
    }

    #[test]
    fn hub_client_creates_without_token() {
        let hub = HfHub::new(None, PathBuf::from("/tmp/hf_test"));
        assert!(hub.is_ok());
    }

    #[test]
    fn hub_client_creates_with_token() {
        let hub = HfHub::new(Some("hf_test_token"), PathBuf::from("/tmp/hf_test"));
        assert!(hub.is_ok());
    }
}
