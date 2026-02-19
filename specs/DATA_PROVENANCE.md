# neuralSpring — Data Provenance

**Last Updated**: February 19, 2026
**Purpose**: Document all external datasets, APIs, and pre-trained assets used in validation experiments. Every data source must be public, reproducible, and free of access restrictions.

---

## Principles (from wateringHole standards)

1. **Public repositories only** — no proprietary, paywalled, or access-restricted data.
2. **Accession numbers** — every dataset has a persistent identifier (DOI, URL, version).
3. **Reproducible access** — anyone with internet can re-fetch the same data.
4. **Pinned versions** — data snapshots are versioned or cached for determinism.

---

## Dataset Inventory

### Experiment Data

| Experiment | Dataset | Source | Access | License | Notes |
|------------|---------|--------|--------|---------|-------|
| Exp 001 (Surrogate) | Synthetic benchmarks | Generated in-code | N/A — pure math | N/A | Rastrigin, Rosenbrock, Ackley functions. Seed=42 for RNG. |
| Exp 001 (Surrogate) | FAO-56 ET₀ | Computed via airSpring | N/A — equation chain | N/A | Allen et al. (1998) FAO Irrigation & Drainage Paper 56. |
| Exp 002 (Transformer) | Synthetic matrices | Generated in-code | N/A — pure math | N/A | Random Q/K/V matrices. Seed=42 for RNG. |
| Exp 003 (Sequence) | Synthetic Michigan weather | Generated in-code | N/A — synthetic | N/A | Realistic seasonal + noise model. Seed=42. |
| Exp 004 (Transfer) | Synthetic domain data | Generated in-code | N/A — synthetic | N/A | Climate-like feature distributions. Seed=42. |
| Exp 005 (Isomorphic) | Architecture catalog | Curated in-code | N/A | N/A | Architecture metadata from published papers. |

### Study Data

| Study | Dataset | Source | Access | License | Notes |
|-------|---------|--------|--------|---------|-------|
| Study 001 (PINN) | Burgers equation | Analytical solution | N/A — PDE | N/A | Raissi et al. (2019). Initial condition: -sin(πx). |
| Study 002 (DeepONet) | Antiderivative operator | Generated in-code | N/A — analytical | N/A | Lu et al. (2021). Polynomial basis functions. |
| Study 003 (LeNet-5) | MNIST handwritten digits | `torchvision.datasets.MNIST` | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) | CC BY-SA 3.0 | LeCun et al. (1998). 60k train / 10k test. Auto-downloaded by PyTorch. |
| Study 004 (LSTM ERA5) | ERA5 reanalysis | Open-Meteo Archive API | [archive-api.open-meteo.com](https://archive-api.open-meteo.com/v1/archive) | CC BY 4.0 | ECMWF Copernicus Climate Data Store. East Lansing, MI (42.73°N, 84.48°W). 2020-01-01 to 2023-12-31. Variables: temperature_2m_max/min, precipitation_sum, wind_speed_10m_max, shortwave_radiation_sum. |
| Study 005 (Quantized) | Synthetic ET₀ data | Generated in-code | N/A — synthetic | N/A | Same FAO-56 chain as Exp 001 with random weather inputs. |

---

## External APIs

| API | URL | Auth | Rate Limit | Fallback |
|-----|-----|------|------------|----------|
| Open-Meteo Archive | `https://archive-api.open-meteo.com/v1/archive` | None (free) | Reasonable use | Cached `.npz` file in `control/lstm_weather/` |

---

## Pre-trained Models / Weights

None. All models are trained from scratch during validation. This is by design — neuralSpring validates the training pipeline, not pre-trained checkpoints.

---

## Caching Strategy

| Data | Cache Location | Format | Regeneration |
|------|---------------|--------|-------------|
| ERA5 weather | `control/lstm_weather/era5_east_lansing_daily.npz` | NumPy compressed | Re-fetch from Open-Meteo API |
| MNIST | `~/.cache/torchvision/` (system default) | PyTorch dataset | Re-download from yann.lecun.com |
| Synthetic data | Not cached | N/A | Regenerated from seed each run |

---

## References

1. Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *Crop evapotranspiration — Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56.
2. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). *Physics-informed neural networks*. Journal of Computational Physics, 378, 686-707. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
3. Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E. (2021). *Learning nonlinear operators via DeepONet*. Nature Machine Intelligence, 3, 218-229. DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)
4. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278-2324. DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
5. Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., Hochreiter, S. (2021). *Rainfall–runoff prediction at multiple timescales with a single Long Short-Term Memory network*. HESS, 25, 2045-2062. DOI: [10.5194/hess-25-2045-2021](https://doi.org/10.5194/hess-25-2045-2021)
6. Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022.
7. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023.

---

## Compliance

- **AGPL-3.0**: All datasets are compatible with AGPL-3.0 distribution.
- **No PII**: No datasets contain personally identifiable information.
- **No proprietary data**: All sources are public and free.
- **Sovereignty**: All data can be independently fetched by any user without institutional access.
