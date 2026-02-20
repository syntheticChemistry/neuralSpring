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
| Exp 003 (Sequence) | ERA5 reanalysis (East Lansing MI) | Open-Meteo Archive API | [archive-api.open-meteo.com](https://archive-api.open-meteo.com/v1/archive) | CC BY 4.0 | 2020–2023 daily tmax/tmin/precip/wind/humidity. Synthetic fallback if API unavailable. |
| Exp 004 (Transfer) | ERA5 reanalysis (3 cities) | Open-Meteo Archive API | [archive-api.open-meteo.com](https://archive-api.open-meteo.com/v1/archive) | CC BY 4.0 | East Lansing MI, Las Cruces NM, Davis CA. 2020–2023 daily tmax/tmin/rhmax/rhmin/wind/solar. ET₀ computed via FAO-56. Synthetic fallback. |
| Exp 005 (Isomorphic) | Architecture catalog | Curated in-code | N/A | N/A | Architecture metadata from published papers. |

### Study Data

| Study | Dataset | Source | Access | License | Notes |
|-------|---------|--------|--------|---------|-------|
| Study 001 (PINN) | Burgers equation + Raissi ref | Analytical + paper data | [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs) | MIT | Raissi et al. (2019). Cole-Hopf analytical + paper Table 1 L2≈6.7e-4. |
| Study 002 (DeepONet) | Antiderivative + Lu ref | Analytical + paper metrics | [lululxvi/deeponet](https://github.com/lululxvi/deeponet) | Apache-2.0 | Lu et al. (2021). Polynomial antiderivatives + paper MSE≈9.27e-7. |
| Study 003 (LeNet-5) | MNIST handwritten digits | `torchvision.datasets.MNIST` | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) | CC BY-SA 3.0 | LeCun et al. (1998). 60k train / 10k test. Auto-downloaded by PyTorch. |
| Study 004 (LSTM ERA5) | ERA5 reanalysis | Open-Meteo Archive API | [archive-api.open-meteo.com](https://archive-api.open-meteo.com/v1/archive) | CC BY 4.0 | ECMWF Copernicus Climate Data Store. East Lansing, MI (42.73°N, 84.48°W). 2020-01-01 to 2023-12-31. Variables: temperature_2m_max/min, precipitation_sum, wind_speed_10m_max, shortwave_radiation_sum. |
| Study 005 (Quantized) | ERA5 reanalysis (East Lansing MI) | Open-Meteo Archive API | [archive-api.open-meteo.com](https://archive-api.open-meteo.com/v1/archive) | CC BY 4.0 | Real ERA5 weather → FAO-56 ET₀ targets. Synthetic fallback if API unavailable. |

### Paper Reproduction Data

| Paper | Dataset | Source | Access | License | Notes |
|-------|---------|--------|--------|---------|-------|
| Paper 11 (CD Evolution) | NK fitness landscapes | Generated in-code | N/A — computational model | N/A | Iram/Dolson (2020) Nature Physics. Wright-Fisher dynamics, N=5, K=2-4. Paper values: speedup ~2-5×. |
| Paper 12 (MODES) | NK + random walk systems | Generated in-code + paper CSVs | [emilydolson/MODES-toolbox-paper](https://github.com/emilydolson/MODES-toolbox-paper) | MIT | Dolson et al. (2019). NK landscape and Avida digital organism CSVs available. |
| Paper 13 (Eco Dynamics) | Multi-niche NK landscape | Generated in-code | N/A — computational model | N/A | Dolson & Ofria (2018) GECCO. Multi-niche Gaussian fitness, N=20 loci, 1-8 niches. |
| Paper 14 (Directed Evo) | Multi-objective landscape | Generated in-code | N/A — computational model | N/A | Dolson et al. (2022) eLife. 5 selection algorithms, 4-objective fitness. |
| Paper 16 (HMM Phylo) | HMM transition/emission | Generated in-code | N/A — computational model | N/A | Liu et al. (2014) PLoS Comp Bio. 2-state weather HMM + 4-state phylo HMM. |
| Paper 19 (Game Theory) | Payoff matrices + QS model | Generated in-code | N/A — computational model | N/A | Bruger & Waters (2018) AEM. PD, snowdrift, QS cooperation, spatial PD. |

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
| ERA5 weather (all experiments) | `data/weather/open_meteo_*.npz` | NumPy compressed | Re-fetch from Open-Meteo API via `control/shared/open_meteo.py` |
| ERA5 weather (legacy LSTM) | `control/lstm_weather/era5_east_lansing_daily.npz` | NumPy compressed | Re-fetch from Open-Meteo API |
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
8. Iram, S., Dolson, E., Chiel, J., Hu, J., Nicholson, C., Ponce, E., Butts, C.T., Raman, R., Ohno, C.L. (2020). *Controlling the speed and trajectory of evolution with counterdiabatic driving*. Nature Physics, 17, 135-142. DOI: [10.1038/s41567-020-0989-3](https://doi.org/10.1038/s41567-020-0989-3)
9. Dolson, E.L., Vostinar, A.E., Wiser, M.J., Ofria, C. (2019). *The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems*. Artificial Life, 25(1), 50-73. DOI: [10.1162/artl_a_00280](https://doi.org/10.1162/artl_a_00280)
10. Dolson, E.L. & Ofria, C. (2018). *Ecological Theory Provides Insights about Evolutionary Computation*. GECCO '18 Companion, pp 105-106. DOI: [10.1145/3205651.3205780](https://doi.org/10.1145/3205651.3205780)
11. Dolson, E.L., Banzhaf, W., Ofria, C. (2022). *Artificial selection methods from evolutionary computing show promise for directed evolution of microbes*. eLife, 11, e79665. DOI: [10.7554/eLife.79665](https://doi.org/10.7554/eLife.79665)
12. Liu, L., Yu, L., Kubatko, L., Pearl, D.K., Edwards, S.V. (2014). *Coalescent methods for estimating phylogenetic trees*. PLoS Computational Biology, 10(4), e1003649. DOI: [10.1371/journal.pcbi.1003649](https://doi.org/10.1371/journal.pcbi.1003649)
13. Bruger, E. & Waters, C.M. (2018). *Maximizing Growth Yield and Dispersal via Quorum Sensing Promotes Cooperation in Vibrio Bacteria*. Applied and Environmental Microbiology, 84(6), e00402-18. DOI: [10.1128/AEM.00402-18](https://doi.org/10.1128/AEM.00402-18)

---

## Compliance

- **AGPL-3.0**: All datasets are compatible with AGPL-3.0 distribution.
- **No PII**: No datasets contain personally identifiable information.
- **No proprietary data**: All sources are public and free.
- **Sovereignty**: All data can be independently fetched by any user without institutional access.
