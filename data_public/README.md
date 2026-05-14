# Public, de-identified release of the WheatGPC dataset

This folder contains the processed feature tables used to fit, cross-validate, and report the results of the manuscript

> **A Phenology-Aligned Temporal Framework Improves Satellite-Based Field-Level Wheat Grain Protein Prediction**

de-identified for public release.

## What was changed relative to the internal `notebooks/data/processed/` mirror

| Internal column | Public release |
|---|---|
| `centroid_lat`, `centroid_lon` | **Dropped.** Exact field-centroid coordinates are not redistributed. |
| `field_key` (partner-issued, e.g. `gmi_…`) | **Replaced** with an anonymous integer (1 .. N), consistent across every file in this folder. |
| `county`, `state` | Retained when known. Most records (≈ 93 %) carry a U.S.\ county + state attribution. |
| `county_fips` | **Added** — 5-digit Census GEOID derived from `(county, state)` by name lookup against the cached `cb_2022_us_county_500k` shapefile. |

The remaining columns (vegetation-index aggregates, meteorological aggregates, soil and topographic features, `protein_pct`, `yield_bu_ac`, etc.) are unchanged.

## How it was generated

```bash
python scripts/deidentify_public_release.py
```

The script is fully deterministic: the anonymous-integer mapping is built from the sorted union of all original keys, so re-running on the same source data reproduces identical outputs.

## Caveats

- A small fraction (~7 %) of records have no county/state attribution in the source data; their `county`, `state`, and `county_fips` fields are left empty.
- All county-level information should be treated as approximate: a field's centroid may fall within a county whose typical management is not representative of the specific field.
- Researchers requiring access at finer spatial granularity may request it from the authors subject to the relevant data-sharing terms (see *Data and code availability* in the paper).
