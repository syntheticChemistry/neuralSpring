# NestGate Provider Plan: HuggingFaceProvider and ERA5Provider

**Date**: February 26, 2026 (Session 81)
**Author**: Kevin Mok
**Status**: PLANNED
**Spring**: neuralSpring → phase1/nestgate integration
**License**: AGPL-3.0-or-later

---

## Purpose

neuralSpring's Tier 2 and Tier 3 extension experiments require external data:
public model weights from HuggingFace and climate reanalysis from ERA5. Rather
than ad-hoc downloads, these should flow through NestGate's existing provider
infrastructure — content-addressed, Blake3-hashed, provenance-tracked, compressed
via the same pipeline that serves NCBI genomic data.

## Existing Pattern: NCBILiveProvider

```
nestgate-core/src/data_sources/providers/live_providers/ncbi_live_provider.rs
```

| Component | Implementation |
|-----------|---------------|
| **Struct** | `NCBILiveProvider { http_provider, api_key, email }` |
| **Trait** | `DataCapability` + `GenomeDataCapability` |
| **API** | `esearch()` → `esummary()` → `efetch()` |
| **Storage** | Blake3 content-addressed, 2-level sharding |
| **Registration** | `UniversalDataAdapter::register_provider()` |
| **Config** | `HttpProviderConfigBuilder::new(base_url, capability_type)` |
| **Factory** | `NCBIProviderFactory::create_from_env()` |

---

## Provider 1: HuggingFaceLiveProvider

### What It Serves

| Data | Consumer | License | Size |
|------|----------|---------|------|
| GPT-2 117M weights | Exp-050, nS-01 spectral analysis | MIT (OpenAI) | ~500MB |
| DistilBERT 66M weights | Exp-051, cross-architecture fingerprint | Apache-2.0 (HuggingFace) | ~250MB |
| ResNet-18 11M weights | Exp-051, CNN spectral comparison | BSD (PyTorch/Meta) | ~45MB |
| ViT-small 22M weights | Exp-051, vision transformer spectral analysis | Apache-2.0 (Google) | ~90MB |
| GPT-2 fine-tune checkpoints | Tier 3, training dynamics at scale | MIT | ~2GB |

### API Mapping

HuggingFace Hub API (https://huggingface.co/api):

| NCBILiveProvider | HuggingFaceLiveProvider | HF API Endpoint |
|------------------|-------------------------|-----------------|
| `esearch()` | `search_models()` | `GET /api/models?search={query}&filter={tag}` |
| `esummary()` | `get_model_info()` | `GET /api/models/{model_id}` |
| `efetch()` | `download_weights()` | `GET /{model_id}/resolve/{revision}/{filename}` |

### Struct

```rust
pub struct HuggingFaceLiveProvider {
    http_provider: UniversalHttpProvider,
    api_token: Option<String>,  // HF_TOKEN env var (optional for public models)
    cache_dir: PathBuf,         // NestGate content-addressed storage root
}
```

### Trait Implementation

```rust
impl DataCapability for HuggingFaceLiveProvider {
    fn capability_type(&self) -> &str { "model_data" }

    async fn can_handle(&self, request: &DataRequest) -> Result<bool> {
        Ok(request.capability_type == "model_data"
            && request.parameters.get("source") == Some(&"huggingface".into()))
    }

    async fn execute_request(&self, request: &DataRequest) -> Result<DataResponse> {
        let model_id = request.parameters.get("model_id")
            .ok_or_else(|| anyhow!("model_id required"))?;
        let filename = request.parameters.get("filename")
            .unwrap_or(&"model.safetensors".to_string());

        let bytes = self.download_weights(model_id, filename).await?;
        let hash = self.store_content_addressed(&bytes, model_id, filename).await?;

        Ok(DataResponse {
            data: bytes,
            metadata: HashMap::from([
                ("content_hash".into(), hex::encode(hash)),
                ("model_id".into(), model_id.clone()),
                ("source".into(), "huggingface".into()),
            ]),
            source_info: SourceInfo {
                provider_type: "model_data".into(),
                provider_name: "HuggingFaceLiveProvider".into(),
                license: self.get_model_license(model_id).await?,
            },
        })
    }
}

impl ModelDataCapability for HuggingFaceLiveProvider {
    async fn search_models(&self, query: &str) -> Result<Vec<ModelResult>> { ... }
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> { ... }
}
```

### Factory

```rust
pub struct HuggingFaceProviderFactory;

impl HuggingFaceProviderFactory {
    pub fn create_from_env() -> Result<Arc<HuggingFaceLiveProvider>> {
        let token = std::env::var("HF_TOKEN").ok();
        let config = HttpProviderConfigBuilder::new(
            "https://huggingface.co".into(),
            "model_data".into(),
        )
        .with_timeout(300)  // Large model downloads
        .with_metadata("provider".into(), "huggingface".into())
        .build();

        Ok(Arc::new(HuggingFaceLiveProvider {
            http_provider: UniversalHttpProvider::new(config)?,
            api_token: token,
            cache_dir: PathBuf::from("/data/nestgate/huggingface"),
        }))
    }
}
```

### Provenance Tracking

Each downloaded model gets a provenance record in NestGate metadata:

```json
{
  "source": "huggingface",
  "model_id": "openai-community/gpt2",
  "revision": "main",
  "sha256": "...",
  "blake3": "...",
  "license": "MIT",
  "author": "OpenAI",
  "parameters": 117000000,
  "download_date": "2026-02-26T12:00:00Z",
  "neuralspring_experiment": "Exp-051"
}
```

---

## Provider 2: ERA5LiveProvider

### What It Serves

| Data | Consumer | License | Size |
|------|----------|---------|------|
| ERA5 hourly surface (Michigan, 4 years) | Exp-050, LSTM training | CC BY 4.0 (Copernicus/ECMWF) | ~500MB |
| ERA5 extended (KBS, 30 years) | gen3 Sub-06 (No-till), Exp-052 | CC BY 4.0 | ~3GB |
| ERA5 global subset (validation) | Cross-climate transfer learning | CC BY 4.0 | ~1GB |

### API Mapping

Open-Meteo API (https://open-meteo.com/en/docs) — free, no API key:

| NCBILiveProvider | ERA5LiveProvider | Open-Meteo Endpoint |
|------------------|------------------|---------------------|
| `esearch()` | `search_variables()` | `GET /v1/era5?latitude={}&longitude={}&hourly={}` |
| `esummary()` | `get_metadata()` | Response headers + variable descriptions |
| `efetch()` | `download_era5()` | `GET /v1/era5?start_date={}&end_date={}` |

### Struct

```rust
pub struct ERA5LiveProvider {
    http_provider: UniversalHttpProvider,
    cache_dir: PathBuf,
}
```

### Trait Implementation

```rust
impl DataCapability for ERA5LiveProvider {
    fn capability_type(&self) -> &str { "research_data" }

    async fn can_handle(&self, request: &DataRequest) -> Result<bool> {
        Ok(request.capability_type == "research_data"
            && request.parameters.get("source") == Some(&"era5".into()))
    }

    async fn execute_request(&self, request: &DataRequest) -> Result<DataResponse> {
        let latitude = request.parameters.get("latitude")
            .ok_or_else(|| anyhow!("latitude required"))?;
        let longitude = request.parameters.get("longitude")
            .ok_or_else(|| anyhow!("longitude required"))?;
        let start_date = request.parameters.get("start_date")
            .ok_or_else(|| anyhow!("start_date required"))?;
        let end_date = request.parameters.get("end_date")
            .ok_or_else(|| anyhow!("end_date required"))?;
        let variables = request.parameters.get("variables")
            .unwrap_or(&"temperature_2m,relative_humidity_2m".to_string());

        let data = self.download_era5(latitude, longitude, start_date, end_date, variables).await?;
        let hash = self.store_content_addressed(&data).await?;

        Ok(DataResponse {
            data,
            metadata: HashMap::from([
                ("content_hash".into(), hex::encode(hash)),
                ("source".into(), "era5_open_meteo".into()),
                ("license".into(), "CC-BY-4.0".into()),
                ("latitude".into(), latitude.clone()),
                ("longitude".into(), longitude.clone()),
                ("date_range".into(), format!("{start_date}..{end_date}")),
            ]),
            source_info: SourceInfo {
                provider_type: "research_data".into(),
                provider_name: "ERA5LiveProvider".into(),
                license: "CC-BY-4.0".into(),
            },
        })
    }
}

impl ResearchDataCapability for ERA5LiveProvider {
    async fn search_research(&self, query: &str) -> Result<Vec<ResearchResult>> { ... }
    async fn get_research_data(&self, research_id: &str) -> Result<ResearchData> { ... }
}
```

### Factory

```rust
pub struct ERA5ProviderFactory;

impl ERA5ProviderFactory {
    pub fn create() -> Result<Arc<ERA5LiveProvider>> {
        let config = HttpProviderConfigBuilder::new(
            "https://archive-api.open-meteo.com".into(),
            "research_data".into(),
        )
        .with_timeout(120)
        .with_metadata("provider".into(), "era5_open_meteo".into())
        .with_metadata("license".into(), "CC-BY-4.0".into())
        .build();

        Ok(Arc::new(ERA5LiveProvider {
            http_provider: UniversalHttpProvider::new(config)?,
            cache_dir: PathBuf::from("/data/nestgate/era5"),
        }))
    }
}
```

---

## Registration

Both providers register into NestGate's `UniversalDataAdapter`:

```rust
let ncbi = NCBIProviderFactory::create_from_env()?;
let hf = HuggingFaceProviderFactory::create_from_env()?;
let era5 = ERA5ProviderFactory::create()?;

let adapter = UniversalDataAdapterBuilder::new()
    .with_provider(ncbi)
    .with_provider(hf)
    .with_provider(era5)
    .build();
```

JSON-RPC capability announcement:

```json
[
  { "capability": "genome_data", "provider": "NCBILiveProvider" },
  { "capability": "model_data", "provider": "HuggingFaceLiveProvider" },
  { "capability": "research_data", "provider": "ERA5LiveProvider" }
]
```

---

## Storage Architecture

```
Westgate (76TB ZFS)
  /data/nestgate/
  ├── ncbi/           # Existing: NCBI sequences, SRA data
  │   ├── data/{shard}/{shard}/{blake3_hash}
  │   └── metadata/{shard}/{shard}/{blake3_hash}.json
  ├── huggingface/    # New: Model weights
  │   ├── data/{shard}/{shard}/{blake3_hash}
  │   └── metadata/{shard}/{shard}/{blake3_hash}.json
  └── era5/           # New: Climate reanalysis
      ├── data/{shard}/{shard}/{blake3_hash}
      └── metadata/{shard}/{shard}/{blake3_hash}.json
```

All providers share:
- Blake3 content-addressed hashing
- 2-level shard directories
- Zstd compression (2.5x ratio)
- Deduplication on hash match
- Atomic writes (temp + rename)

---

## Implementation Order

1. **HuggingFaceLiveProvider** (Tier 2 gate — needed for Exp-051 cross-architecture)
   - Implement struct + `DataCapability` + `ModelDataCapability`
   - Wire into `UniversalDataAdapter`
   - Test: download GPT-2 weights, verify Blake3 hash, check dedup on re-download
   - Estimated effort: 1-2 sessions

2. **ERA5LiveProvider** (Tier 2 gate — needed for Exp-050 extended training)
   - Implement struct + `DataCapability` + `ResearchDataCapability`
   - Wire into `UniversalDataAdapter`
   - Test: download Michigan ERA5 data, verify against existing Python pipeline
   - Estimated effort: 1 session

3. **Model checkpoint storage** (Tier 1 enhancement — useful but not blocking)
   - Extend HuggingFaceProvider with `store_checkpoint()` for our own training runs
   - Content-addressed checkpoints with training metadata (epoch, loss, architecture)
   - Estimated effort: 0.5 session

---

## Sovereignty Notes

- HuggingFace: Public models only (MIT/Apache/BSD). No proprietary model downloads.
  Token optional — only needed for private repos (which we don't use)
- ERA5: Open-Meteo is a free proxy for Copernicus Climate Data Store. CC-BY-4.0.
  No API key required. No institutional access needed
- Both providers cache locally on NestGate ZFS — once downloaded, no external
  dependency for subsequent access. Full air-gap capability after initial fetch
- AGPL-3.0 license on all provider code
