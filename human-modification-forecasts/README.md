# Probabilistic Forecasts of Global Human Modification, 2025–2040

Global probabilistic forecasts of cumulative **Human Modification (HM)** of terrestrial ecosystems at ~1 km resolution for 2025, 2030, 2035 and 2040, together with the observed HM record for 1990–2020 on the same grid. The data are produced by [The Nature Conservancy](https://www.nature.org/) and hosted by the [AWS Open Data Sponsorship Program](https://aws.amazon.com/opendata/).

- **Registry of Open Data entry:** https://registry.opendata.aws/tnc-human-modification-forecasts
- **Tutorial notebook:** [`get-to-know-a-dataset.ipynb`](get-to-know-a-dataset.ipynb)
- **Paper:** Moncrieff et al. (2026), *Probabilistic projections of human modification of the terrestrial biosphere from 2020 to 2040*. https://doi.org/10.32942/X2S11G
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Contact:** glenn.moncrieff@tnc.org, or [open an issue](https://github.com/GMoncrieff/open-data-examples/issues)

---

## Contents

1. [What is Human Modification?](#what-is-human-modification)
2. [How the forecasts were made](#how-the-forecasts-were-made)
3. [Accessing the data](#accessing-the-data)
4. [Bucket layout](#bucket-layout)
5. [Data dictionary](#data-dictionary)
6. [Grid and encoding](#grid-and-encoding)
7. [Working with the forecast distribution](#working-with-the-forecast-distribution)
8. [Aggregating over regions: do's and don'ts](#aggregating-over-regions-dos-and-donts)
9. [Forecast performance](#forecast-performance)
10. [Limitations](#limitations)
11. [Related products](#related-products)
12. [Citation](#citation)

---

## What is Human Modification?

Human Modification (HM) is a continuous, unitless index of the degree to which a landscape has been modified by people. It ranges from **0** (no detectable human modification) to **1** (complete modification). HM combines 16 anthropogenic threats grouped into eight themes: agriculture, built environments, extraction, biological resource use, human accessibility, natural systems modification, pollution, and transportation. These map to seven of the 11 top-level direct threat categories in the IUCN threat classification. Each threat is mapped from globally consistent remote-sensing and ancillary data, scaled to 0–1, and combined cumulatively, weighted by its expected intensity of impact ([Theobald et al. 2025](https://doi.org/10.1038/s41597-025-04892-2)).

Commonly used thresholds (following Kennedy et al. and Theobald et al.):

| HM | Class |
|---|---|
| ≤ 0.1 | Natural / low modification |
| 0.1 – 0.4 | Moderately modified |
| > 0.4 | Highly modified |

Unlike categorical land-cover products, HM captures **intensification and degradation within** a land-cover class, such as new roads, pollution or human intrusion into intact forest, as well as conversion between classes.

## How the forecasts were made

A convolutional long short-term memory (convLSTM) network was trained on the 1990–2020 HM time series (resampled from 300 m to 1 km). Its inputs are sequences of three consecutive HM maps, the eight HM threat themes, and dynamic and static covariates:

- population density and GDP
- climate: mean annual temperature, mean annual precipitation, minimum temperature of the coldest month
- elevation, slope and aspect
- strictly protected areas and other protected areas
- the Development Suitability Index
- neighbourhood measures of past HM and HM change
- a learned spherical-harmonic location encoding

Rather than a single value, the model predicts each pixel's **full distribution** of future HM. It is trained by minimising the continuous ranked probability score (CRPS), and its output is a monotone piecewise-linear quantile function. Forecasts for 2025–2040 were made from a model trained on all available data, using inputs from 2010–2020. Full methods are in the [paper](https://doi.org/10.32942/X2S11G) and the [model code](https://github.com/GMoncrieff/spatio-temporal).

The forecasts are **conditional on the continuation of historical relationships and dynamics**. They are not scenarios of alternative policy or socioeconomic futures.

## Accessing the data

The data are in a public S3 bucket in `us-west-2`. No AWS account or credentials are needed.

| | |
|---|---|
| Bucket | `s3://tnc-human-modification` |
| Region | `us-west-2` |
| Repository prefix | `hm_probability_forecast` |
| Format | [Icechunk](https://icechunk.io) repository of [Zarr v3](https://zarr.dev) arrays |
| Branch | `main` |

### Python (recommended)

```bash
pip install "icechunk>=1.0" "zarr>=3" xarray
```

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="tnc-human-modification",
    prefix="hm_probability_forecast",
    region="us-west-2",
    anonymous=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="main")

forecast = xr.open_zarr(session.store, group="hm_prob_forecast", consolidated=False, chunks=None)
observed = xr.open_zarr(session.store, group="hm_observed", consolidated=False, chunks=None)

# Full forecast distribution for one pixel, all four years
px = forecast.hm.sel(latitude=0.05, longitude=37.05, method="nearest").load()
```

`chunks=None` gives lazy arrays without dask, which suits regional work. For continental or global analysis, use `chunks={}` to get dask arrays aligned to the stored chunks, and run close to the data in `us-west-2` (EC2, SageMaker AI, or a Dask cluster).

### Browsing the raw objects

The objects under `hm_probability_forecast/` are Icechunk's internal storage: repository metadata, `snapshots/`, `manifests/`, `chunks/` and `transactions/`, with content-addressed names. They cannot be used directly; always open the repository with an Icechunk client.

```bash
aws s3 ls --no-sign-request s3://tnc-human-modification/hm_probability_forecast/
```

### Not using Python?

Single-band GeoTIFFs of summary layers (mean, 2.5th and 97.5th percentiles, and the probabilities of exceeding HM 0.1 and 0.4 for each forecast year, plus observed HM 1990–2020) are on [Zenodo](https://doi.org/10.5281/zenodo.19700998) for use in any GIS. An interactive browser for the full dataset is on the [Earthmover Data Marketplace](https://app.earthmover.io/marketplace/6abb60fba504351ef2a29c6f), and a 3D globe explorer is at [gmoncrieff.github.io/hm_forecasts_app](https://gmoncrieff.github.io/hm_forecasts_app/).

## Bucket layout

```
s3://tnc-human-modification/
└── hm_probability_forecast/          Icechunk repository
    │                                 (internal objects; open with icechunk)
    │
    │   Zarr hierarchy as seen through icechunk:
    ├── hm_prob_forecast/             group: forecast quantile functions, 2025–2040
    │   ├── hm                        (year, percentile, latitude, longitude)  uint16
    │   ├── year                      (year)        int32
    │   ├── percentile                (percentile)  float64
    │   ├── latitude                  (latitude)    float64
    │   ├── longitude                 (longitude)   float64
    │   └── crs                       scalar grid-mapping variable
    └── hm_observed/                  group: observed HM, 1990–2020
        ├── hm                        (year, latitude, longitude)  uint16
        ├── year                      (year)        int32
        ├── latitude                  (latitude)    float64
        ├── longitude                 (longitude)   float64
        └── crs                       scalar grid-mapping variable
```

## Data dictionary

### Group `hm_prob_forecast`

| Variable | Dimensions | Shape | Stored type | Decoded type / units | Description |
|---|---|---|---|---|---|
| `hm` | year, percentile, latitude, longitude | 4 × 64 × 17,111 × 40,000 | uint16 | float, unitless (0–1) | Forecast HM at each quantile level, i.e. the pixel's quantile function. `NaN` (after decoding) over ocean, permanent inland water and Antarctica. |
| `year` | year | 4 | int32 | year | Target year: 2025, 2030, 2035, 2040 (forecast horizons of +5, +10, +15, +20 years from 2020). |
| `percentile` | percentile | 64 | float64 | percent | Quantile level in percent, from 0.0248 to 99.99. 2.5, 50 and 97.5 are exact levels. Levels are denser in the tails (see below). |
| `latitude` | latitude | 17,111 | float64 | degrees_north | Pixel-centre latitude, **descending** from ~83.99° N to ~70.00° S. |
| `longitude` | longitude | 40,000 | float64 | degrees_east | Pixel-centre longitude, ascending from ~−179.995° to ~179.995°. |
| `crs` | – | scalar | int32 | – | CF grid-mapping variable (EPSG:4326, WKT and GDAL GeoTransform in attributes). |

### Group `hm_observed`

| Variable | Dimensions | Shape | Stored type | Decoded type / units | Description |
|---|---|---|---|---|---|
| `hm` | year, latitude, longitude | 7 × 17,111 × 40,000 | uint16 | float, unitless (0–1) | Observed cumulative HM from [Theobald et al. (2025)](https://doi.org/10.1038/s41597-025-04892-2), average-resampled from 300 m to the forecast grid and masked identically. |
| `year` | year | 7 | int32 | year | 1990, 1995, 2000, 2005, 2010, 2015, 2020. |
| `latitude`, `longitude`, `crs` | | | | | Identical to the forecast group. |

### Percentile levels

The 64 quantile levels (percent) are:

```
0.0248, 0.0383, 0.0584, 0.0878, 0.130, 0.191, 0.276, 0.395, 0.557, 0.775,
1.07, 1.45, 1.94, 2.5, 2.57, 3.36, 4.35, 5.55, 7.00, 8.73,
10.8, 13.1, 15.8, 18.8, 22.1, 25.8, 29.8, 34.0, 38.4, 43.0,
47.6, 50, 52.4, 57.0, 61.6, 66.0, 70.2, 74.2, 77.9, 81.2,
84.2, 86.9, 89.2, 91.3, 93.0, 94.5, 95.7, 96.6, 97.4, 97.5,
98.1, 98.6, 98.9, 99.2, 99.4, 99.6, 99.7, 99.8, 99.87, 99.91,
99.94, 99.96, 99.975, 99.99
```

Always read exact values from the `percentile` coordinate rather than typing them. Select exact levels with `.sel(percentile=50.0)`, or the nearest level with `.sel(percentile=25, method="nearest")`.

## Grid and encoding

| Property | Value |
|---|---|
| CRS | EPSG:4326 (WGS 84 geographic) |
| Resolution | 0.009° (~1 km at the equator) |
| Extent | 180° W – 180° E, 84.00° N – 70.00° S (land only) |
| Grid size | 40,000 columns × 17,111 rows |
| Storage type | uint16 |
| Scale factor | 1/65535 (HM = stored value × 1/65535) |
| Fill value | 65535 (ocean, permanent inland water, Antarctica) |
| Valid range | 0 – 65534, so the largest representable HM is 65534/65535 ≈ 0.99998 |
| Compression | Blosc (zstd, level 5, byte shuffle) |
| Chunks (forecast) | 1 year × 64 percentiles × 256 lat × 320 lon (~10 MB uncompressed) |
| Chunks (observed) | 1 year × 256 lat × 320 lon |
| Shards | 1024 lat × 1280 lon (4 × 4 chunks per stored object) |
| Size on disk | ~70 GB compressed for the forecast group (~350 GB uncompressed) |

The scale factor and fill value are stored as CF attributes, and **xarray decodes them automatically**: `hm` arrives as floating-point HM with `NaN` where there is no data. If you read with `zarr` directly, apply the decoding yourself:

```python
raw = zarr_array[...]                       # uint16
hm = np.where(raw == 65535, np.nan, raw / 65535.0)
```

**Chunking rationale.** The percentile axis is never split across chunks, so a single read returns the complete distribution for every pixel in a 256 × 320 window. Reading one pixel, or a small region, for one year touches one or a few objects. Reading one percentile for the whole globe is expensive (roughly 17 GB of compressed chunks for one year), because every chunk must be fetched. Design analyses to work region by region and use all percentiles from each read.

**Statistics and areas.** The grid is geographic, so pixel area shrinks with latitude. Weight by pixel area, or reproject to an equal-area grid, before reporting areas or regional averages. The paper uses Lambert Cylindrical Equal Area (EPSG:6933). The area of a 0.009° cell centred at latitude φ is R² · Δλ · (sin(φ + Δ/2) − sin(φ − Δ/2)), with Δ = Δλ = 0.009° in radians.

## Working with the forecast distribution

Each pixel's forecast is a monotone, piecewise-linear quantile function *Q(u)* defined at the 64 stored levels *u*. Linear interpolation between levels is consistent with how the model represents the distribution. Useful derived quantities:

| Quantity | How to compute | Notes |
|---|---|---|
| Median / any percentile | `hm.sel(percentile=p)` (exact) or interpolate between neighbouring levels | |
| 95% prediction interval | `hm.sel(percentile=[2.5, 97.5])` | |
| Mean (expected HM) | ∫₀¹ Q(u) du, e.g. trapezoid over the 64 levels extended flat to u = 0 and 1 | Corresponds to the "mean" layer on Zenodo |
| P(HM > t) | Invert the quantile function: find the levels bracketing *t* and interpolate *u* linearly; P = 1 − u | Corresponds to the "gt01" (t = 0.1) and "gt04" (t = 0.4) layers on Zenodo |
| Random draw | Draw *u* ~ Uniform(0, 1) and interpolate Q at *u* | Draws are independent per pixel; see below |

The [tutorial notebook](get-to-know-a-dataset.ipynb) includes reusable `prob_exceed()` and `quantile_mean()` functions that work on any xarray object with a `percentile` dimension.

Probabilities outside the range of stored levels are clamped to [0.000248, 0.9999]. For example, P(HM > t) is reported as ≤ 0.0001 when *t* exceeds the 99.99th percentile.

## Aggregating over regions: do's and don'ts

Forecast distributions are estimated **independently for each pixel** and do not describe how pixels co-vary. This determines which regional statistics are valid:

| Statistic | Valid? | Why |
|---|---|---|
| Sum or mean of **expected HM** over a region | ✅ | The expectation of a sum is the sum of expectations, regardless of correlation |
| **Expected area** beyond a threshold: Σ pixel area × P(HM > t) | ✅ | Same reason. Two 1 km² pixels each with P = 0.2 contribute 0.4 km² |
| **Area or proportion of pixels** whose *k*-th percentile exceeds a threshold | ✅ | A statement about how widespread a plausible outcome is |
| Sum or mean of a **percentile map** reported as the regional *k*-th percentile | ❌ | Would require every pixel to realise its *k*-th percentile simultaneously |
| Regional totals from **independent per-pixel random draws** | ❌ | Ignores spatial dependence; strongly underestimates regional variance |

The upper (97.5th percentile) forecast is best used as a **precautionary risk screen**. It shows where large increases in HM are plausible, though not the most likely outcome. It is not a worst-case regional total.

## Forecast performance

Out-of-sample performance on spatially held-out tiles (five-fold spatial cross-validation, 1990–2020), from the paper:

| Horizon | +5 yr | +10 yr | +15 yr | +20 yr |
|---|---|---|---|---|
| RMSE (mean forecast) | 0.0133 | 0.0203 | 0.0264 | 0.0312 |
| RMSE skill vs persistence | 0.041 | 0.188 | 0.229 | 0.241 |
| CRPS (full distribution) | 0.0030 | 0.0053 | 0.0073 | 0.0089 |
| CRPS skill vs persistence | 0.155 | 0.261 | 0.299 | 0.299 |
| 50% interval coverage | 0.421 | 0.427 | 0.449 | 0.482 |
| 80% interval coverage | 0.773 | 0.762 | 0.760 | 0.772 |
| 95% interval coverage | 0.950 | 0.917 | 0.953 | 0.955 |
| 99% interval coverage | 0.991 | 0.991 | 0.991 | 0.990 |

The persistence baseline assumes HM stays at its last observed value. Outer intervals are well calibrated. Inner intervals are slightly too narrow, so the forecasts are somewhat over-confident near the median. The mean forecast systematically underpredicts large changes, which is a property of any expected value. Use the full distribution, or exceedance probabilities, to represent rare large changes.

## Limitations

- **Historical continuation.** The model learns from 1990–2020 dynamics and assumes they continue. It cannot anticipate new drivers, such as new mining frontiers, energy transitions, policy shifts or conflict, that are poorly represented in the historical record.
- **Cumulative HM only.** Forecasts are of total HM, not of individual threat components.
- **Horizon.** Forecasts extend 20 years, limited by the length of the historical record. Climate change and socioeconomic scenario uncertainty are not represented and may dominate at longer horizons.
- **Per-pixel distributions.** There is no joint spatial dependence; see [Aggregating over regions](#aggregating-over-regions-dos-and-donts).
- **Protected-area inputs** inherit the limitations of the World Database on Protected Areas: incomplete coverage, boundary and date errors, and uneven application of IUCN management categories. Protected-area masks reflect contemporary rather than historical status.
- **Resolution.** Inputs were resampled to ~1 km. The forecasts suit landscape-to-global planning, not site-level decisions.

## Related products

| Product | Where |
|---|---|
| Summary GeoTIFFs (mean, 2.5th and 97.5th percentiles, P(HM > 0.1), P(HM > 0.4), observed HM) | [Zenodo, doi:10.5281/zenodo.19700998](https://doi.org/10.5281/zenodo.19700998) |
| Historical HM 1990–2020 at 300 m, with threat components | [Zenodo, records/14449495](https://zenodo.org/records/14449495) |
| Interactive data browser (Arraylake) | [Earthmover Data Marketplace](https://app.earthmover.io/marketplace/6abb60fba504351ef2a29c6f) |
| 3D globe explorer | [gmoncrieff.github.io/hm_forecasts_app](https://gmoncrieff.github.io/hm_forecasts_app/) |
| Model code | [github.com/GMoncrieff/spatio-temporal](https://github.com/GMoncrieff/spatio-temporal) |
| Paper figures and tables code | [github.com/GMoncrieff/hm_forecasting_paper](https://github.com/GMoncrieff/hm_forecasting_paper) |

## Citation

If you use these data, please cite:

> Moncrieff, G. R., Voigt, M., Oakleaf, J. R., Theobald, D. M., Walker, K. S., Fitzsimons, J. A., Kiesecker, J., & Kennedy, C. M. (2026). Probabilistic projections of human modification of the terrestrial biosphere from 2020 to 2040. *EcoEvoRxiv*. https://doi.org/10.32942/X2S11G

and, for the historical HM data:

> Theobald, D. M., Oakleaf, J. R., Moncrieff, G., Voigt, M., Kiesecker, J., & Kennedy, C. M. (2025). Global extent and change in human modification of terrestrial ecosystems from 1990 to 2022. *Scientific Data*, 12, 606. https://doi.org/10.1038/s41597-025-04892-2

Funding for this work was provided by The Nature Conservancy and One Earth.

## License

The data are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
