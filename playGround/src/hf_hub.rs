// SPDX-License-Identifier: AGPL-3.0-or-later

//! `HuggingFace` Hub client for downloading model files.
//!
//! Downloads `config.json` and `*.safetensors` files from HF Hub
//! using the REST API. Routes all HTTP through the Tower Atomic stack
//! (Songbird) via IPC — zero direct HTTP dependencies, zero C deps.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::songbird_http::SongbirdHttp;

fn hf_api_base() -> String {
    std::env::var("HF_API_BASE").unwrap_or_else(|_| "https://huggingface.co/api/models".to_string())
}

fn hf_download_base() -> String {
    std::env::var("HF_DOWNLOAD_BASE").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// `HuggingFace` Hub client routed through Tower Atomic (Songbird).
pub struct HfHub {
    http: SongbirdHttp,
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
    ///
    /// Discovers Songbird at runtime for all HTTP operations.
    pub fn new(token: Option<&str>, cache_dir: PathBuf) -> Result<Self> {
        let mut http =
            SongbirdHttp::discover().context("HfHub requires Songbird (Tower Atomic) for HTTP")?;

        if let Some(t) = token {
            http.set_header("Authorization", format!("Bearer {t}"));
        }

        Ok(Self { http, cache_dir })
    }

    /// Fetch model metadata from HF Hub.
    pub async fn model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let url = format!("{}/{model_id}", hf_api_base());
        self.http
            .get_json(&url)
            .await
            .with_context(|| format!("fetching model info for {model_id}"))
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

        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let url = format!("{}/{model_id}/resolve/main/{filename}", hf_download_base());
        log::info!("Downloading {url} -> {}", dest.display());

        let bytes = self
            .http
            .download_to_file(&url, &dest)
            .await
            .with_context(|| format!("downloading {filename} from {model_id}"))?;

        log::info!("Downloaded {bytes} bytes -> {}", dest.display());
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
            let name = filename.as_str();
            let should_download = name == "config.json"
                || name.ends_with(".safetensors")
                || name == "tokenizer.json"
                || name == "tokenizer_config.json";

            if !should_download {
                continue;
            }

            let path = self.download_file(model_id, name).await?;

            if name == "config.json" {
                config_path = Some(path);
            } else if name.ends_with(".safetensors") {
                safetensor_paths.push(path);
            } else if name == "tokenizer.json" {
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
    pub const fn is_complete(&self) -> bool {
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
}
