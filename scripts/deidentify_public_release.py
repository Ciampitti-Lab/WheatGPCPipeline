"""De-identify the per-field datasets for the public release.

The processed datasets in ``notebooks/data/processed/`` contain two
pieces of identifying information that must not be redistributed:

1. ``centroid_lat`` / ``centroid_lon``: exact field-centroid
   coordinates.
2. ``field_key``: 93\xa0% of records carry the original partner-issued
   identifier (``gmi_…``); we replace every key with an anonymous
   integer.

The released dataset retains the existing ``county`` and ``state``
columns and adds a 5-digit Census GEOID (``county_fips``) by name
lookup against the cached ``cb_2022_us_county_500k`` shapefile. The
anonymous-integer mapping is consistent across every file in the
output directory so that downstream joins continue to work.

Run from the repository root::

    python scripts/deidentify_public_release.py
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

import pandas as pd
import geopandas as gpd

REPO     = Path(__file__).resolve().parents[1]
SRC_DIR  = REPO / 'notebooks' / 'data' / 'processed'
OUT_DIR  = REPO / 'data_public' / 'processed'
COUNTY_SHP = REPO / 'data' / 'cache' / 'cb_2022_us_county_500k' / 'cb_2022_us_county_500k.shp'

LAT_COL = 'centroid_lat'
LON_COL = 'centroid_lon'

DROP_COLS = {LAT_COL, LON_COL,
             'state_left', 'state_right',
             'county_left', 'county_right'}


# ───────────────── county FIPS lookup ─────────────────
US_STATE_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
    'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
    'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16',
    'Illinois': '17', 'Indiana': '18', 'Iowa': '19', 'Kansas': '20',
    'Kentucky': '21', 'Louisiana': '22', 'Maine': '23', 'Maryland': '24',
    'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28',
    'Missouri': '29', 'Montana': '30', 'Nebraska': '31', 'Nevada': '32',
    'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35', 'New York': '36',
    'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40',
    'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45',
    'South Dakota': '46', 'Tennessee': '47', 'Texas': '48', 'Utah': '49',
    'Vermont': '50', 'Virginia': '51', 'Washington': '53', 'West Virginia': '54',
    'Wisconsin': '55', 'Wyoming': '56',
}


def load_county_fips_lookup() -> dict[tuple[str, str], str]:
    """Build a ``(county_name, state_name) -> 5-digit GEOID`` lookup."""
    if not COUNTY_SHP.exists():
        sys.exit(f'County shapefile not found at {COUNTY_SHP}')
    counties = gpd.read_file(COUNTY_SHP)
    state_fips_to_name = {v: k for k, v in US_STATE_FIPS.items()}
    counties['state_name'] = counties['STATEFP'].map(state_fips_to_name)
    counties = counties.dropna(subset=['state_name'])
    return {(row['NAME'], row['state_name']): row['GEOID']
            for _, row in counties.iterrows()}


# ───────────────── canonical field_key mapping ─────────────────
def build_field_key_mapping() -> dict[str, int]:
    """Generate a deterministic ``original_field_key -> anonymous_int`` map.

    The mapping is built from the union of every file's ``field_key`` column
    so that the same original key receives the same anonymous integer
    across all output files. Integers are assigned in alphabetical order of
    the (string-cast) original key for full reproducibility.
    """
    keys: set[str] = set()
    for src in SRC_DIR.iterdir():
        if src.suffix.lower() not in ('.parquet', '.csv') or not src.is_file():
            continue
        try:
            df = pd.read_parquet(src) if src.suffix.lower() == '.parquet' \
                else pd.read_csv(src)
        except Exception:
            continue
        if 'field_key' in df.columns:
            keys.update(df['field_key'].dropna().astype(str).tolist())
    return {k: i + 1 for i, k in enumerate(sorted(keys))}


def build_county_lookup(fips_lookup: dict[tuple[str, str], str]
                         ) -> dict[str, tuple[str, str, str]]:
    """Map original ``field_key`` to ``(county, state, county_fips)``.

    The lookup is sourced from ``static_features.parquet``, which is the
    only processed file that carries both the field-level geographic
    attribution and the original keys for every field.
    """
    static_path = SRC_DIR / 'static_features.parquet'
    if not static_path.exists():
        return {}
    df = pd.read_parquet(static_path)
    if 'field_key' not in df.columns or 'county' not in df.columns \
            or 'state' not in df.columns:
        return {}
    lookup: dict[str, tuple[str, str, str]] = {}
    for _, row in df.iterrows():
        key = str(row['field_key'])
        county = str(row['county']).strip()
        state = str(row['state']).strip()
        fips = fips_lookup.get((county, state)) or ''
        lookup[key] = (county, state, fips)
    return lookup


# ───────────────── per-file processing ─────────────────
def deidentify_dataframe(df: pd.DataFrame,
                         key_map: dict[str, int],
                         fips_lookup: dict[tuple[str, str], str],
                         county_lookup: dict[str, tuple[str, str, str]]
                         ) -> pd.DataFrame:
    df = df.copy()

    # 1. If the file lacks county/state but does have field_key, enrich it
    #    from the static-features lookup before anonymising the keys.
    if 'field_key' in df.columns:
        keys_str = df['field_key'].astype(str)
        if 'county' not in df.columns or 'state' not in df.columns:
            triples = [county_lookup.get(k, ('', '', '')) for k in keys_str]
            df['county'] = [t[0] for t in triples]
            df['state'] = [t[1] for t in triples]
            df['county_fips'] = [t[2] for t in triples]

    # 2. Add or recompute county_fips from county/state when not already
    #    set above.
    if ('county' in df.columns and 'state' in df.columns
            and 'county_fips' not in df.columns):
        df['county_fips'] = [
            fips_lookup.get((str(c).strip(), str(s).strip()), '')
            for c, s in zip(df['county'], df['state'])
        ]

    # 3. Anonymise the field_key.
    if 'field_key' in df.columns:
        df['field_key'] = df['field_key'].astype(str).map(key_map).astype('Int64')

    # 4. Drop identifying / leftover columns.
    drop = [c for c in df.columns if c in DROP_COLS]
    if drop:
        df = df.drop(columns=drop)

    # 5. Reorder canonical columns to the front for readability.
    front = [c for c in ('field_key', 'county_fips', 'county', 'state',
                          'protein_pct', 'yield_bu_ac') if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def process_file(src: Path, dst: Path,
                 key_map: dict[str, int],
                 fips_lookup: dict[tuple[str, str], str],
                 county_lookup: dict[str, tuple[str, str, str]]) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if suffix == '.parquet':
        df = pd.read_parquet(src)
        df = deidentify_dataframe(df, key_map, fips_lookup, county_lookup)
        df.to_parquet(dst, index=False)
        action = f'de-identified ({len(df)} rows)'
    elif suffix == '.csv':
        df = pd.read_csv(src)
        df = deidentify_dataframe(df, key_map, fips_lookup, county_lookup)
        df.to_csv(dst, index=False)
        action = f'de-identified ({len(df)} rows)'
    else:
        shutil.copy2(src, dst)
        action = 'copied (binary)'
    return action


def main() -> None:
    if not SRC_DIR.exists():
        sys.exit(f'Source directory {SRC_DIR} does not exist.')

    print('Building county-FIPS lookup...')
    fips_lookup = load_county_fips_lookup()
    print(f'  {len(fips_lookup)} (county, state) pairs indexed.')

    print('Building anonymous field-key mapping...')
    key_map = build_field_key_mapping()
    print(f'  {len(key_map)} unique field keys -> integers 1..{len(key_map)}.')

    print('Building field_key -> county lookup from static_features.parquet...')
    county_lookup = build_county_lookup(fips_lookup)
    print(f'  {len(county_lookup)} field keys carry county/state attribution.')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Persist the mapping (without the original gmi_… keys) for traceability
    # of the anonymous-integer assignment scheme.
    (OUT_DIR / 'field_key_mapping.json').write_text(json.dumps(
        {'description': ('Anonymous field_key integer assignments. The '
                          'original partner-issued keys are intentionally '
                          'omitted from the public release.'),
         'n_fields':    len(key_map)},
        indent=2,
    ))

    files = sorted(SRC_DIR.iterdir())
    print(f'\nProcessing {len(files)} files from {SRC_DIR.relative_to(REPO)}/')
    print(f'           to     {OUT_DIR.relative_to(REPO)}/\n')
    for src in files:
        if not src.is_file():
            continue
        try:
            action = process_file(src, OUT_DIR / src.name,
                                   key_map, fips_lookup, county_lookup)
        except Exception as e:
            action = f'FAILED: {e}'
        print(f'  {src.name:60s}  {action}')

    print('\nDone. Spot-check the contents of data_public/ before publishing.')


if __name__ == '__main__':
    main()
