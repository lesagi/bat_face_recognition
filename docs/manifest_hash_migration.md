# Manifest hash migration — 2026-08-30 layout normalisation

Both species' trees were normalised to one layout,
`<arm>/<variant>/<identity>/`, and the duplicated identity was removed from
13,231 mauritius filenames (`m--<id>--<id>.<frame>.jpg` ->
`m--<id>--f<frame>.jpg`). Rousettus filenames were already correct — its
identities are names, distinct from its video stems — so only its directories
moved.

`Manifest.from_records` digests the file paths, so every manifest hash below
changed. **Identity partitions did not change**: only the `path` column was
rewritten, and the migration verified split membership before and after for
every manifest (43/43 identical).

MLflow runs recorded before this date carry the OLD hash in
`params.manifest_hash` and `params.split.source_manifest_hash`. Resolve them
through this table — hash equality alone will no longer match.

| manifest | old hash | new hash | rows |
|---|---|---|---|
| `mauritius_band_green_manifest.csv` | `c39392ad1087d4c9` | `16f19f085c09bdea` | 279 |
| `mauritius_band_original_manifest.csv` | `1d4087f679f67aff` | `49e82f7f69225a8b` | 279 |
| `mauritius_band_random_manifest.csv` | `5b781fdb55850779` | `6c4f15582239bfcd` | 279 |
| `mauritius_bgonly_original_manifest.csv` | `e4ee5c5920e976b7` | `3a6e622a967d351d` | 520 |
| `mauritius_bgonly_random_manifest.csv` | `8a52e17f0b100087` | `8a775bc8c27187a6` | 520 |
| `mauritius_blur_green_manifest.csv` | `546c71e59b5040df` | `8b642e99e9e8a950` | 520 |
| `mauritius_blur_original_manifest.csv` | `ca65be1cdc1595cc` | `4e9678fd0d19fb15` | 520 |
| `mauritius_blur_random_manifest.csv` | `d94e007cdfecfd63` | `fb6f2f4627633cbf` | 520 |
| `mauritius_green_aligned_224_manifest.csv` | `2af9f650c96c48a3` | `1069102e6514a0ad` | 520 |
| `mauritius_green_aligned_320_manifest.csv` | `e222a6d02ed3c1e2` | `d802f8d2c832a6bb` | 520 |
| `mauritius_green_bg_manifest.csv` | `1ee3e794125fcfdc` | `b9141359a4c78617` | 520 |
| `mauritius_manifest.csv` | `5af031e79b6d20ea` | `a006dda6e7caa644` | 520 |
| `mauritius_occ_roi_centre_manifest.csv` | `c95fb9d3f5c3dba8` | `dbf38a0227f038bb` | 520 |
| `mauritius_occ_roi_matched_manifest.csv` | `2e9186e645bb49b0` | `36e70bdadcb71dad` | 520 |
| `mauritius_occ_roi_none_manifest.csv` | `fd5304b8dfd7a79a` | `3a386064b2692e7d` | 520 |
| `mauritius_occ_roi_periphery_manifest.csv` | `ae8ad80afabfad7a` | `88c9dd64193d8954` | 520 |
| `mauritius_original_aligned_224_manifest.csv` | `3eecf0f5f20cbabb` | `8abc34bfa1570ff9` | 520 |
| `mauritius_original_aligned_320_manifest.csv` | `fe0a04653747c338` | `5c313789d30797a4` | 520 |
| `mauritius_original_bg_manifest.csv` | `547be32545ec3190` | `ed8ea892e92acdfa` | 520 |
| `mauritius_random_aligned_224_manifest.csv` | `2e3da311800d6697` | `7328d5f747a13848` | 520 |
| `mauritius_random_aligned_320_manifest.csv` | `808264df903abe50` | `24153a74cefb3f68` | 520 |
| `mauritius_recrop_green_manifest.csv` | `946483ad11fe8294` | `a2ddf97172ca5c22` | 520 |
| `mauritius_recrop_original_manifest.csv` | `85dd4e9ce4b22bd4` | `b31ee7fb3f70666d` | 520 |
| `mauritius_recrop_random_manifest.csv` | `f469678276fe999a` | `9b78cf639d887520` | 520 |
| `rousettus_band_green_manifest.csv` | `117f81d1ecb49aba` | `92fcb6a866605610` | 616 |
| `rousettus_band_original_manifest.csv` | `b7063f78298ded82` | `92d71a5ad49554d3` | 616 |
| `rousettus_band_random_manifest.csv` | `3387d4fde7480c3d` | `d1a47735d5f80ae0` | 616 |
| `rousettus_bgonly_original_manifest.csv` | `b48ea3fa814348d9` | `0880559f2ee888cd` | 1059 |
| `rousettus_bgonly_random_manifest.csv` | `ec4280d3359739d2` | `047b6048aa40d2a0` | 1059 |
| `rousettus_green_aligned_224_manifest.csv` | `654dd15f87903404` | `62fc32ab443fd381` | 1092 |
| `rousettus_green_aligned_320_manifest.csv` | `88a3a94f1ed3ec6a` | `0f91d0b1c2eae5c2` | 1092 |
| `rousettus_green_bg_intersect_manifest.csv` | `ed5bf795cfc809bc` | `64d968dbf7f88962` | 1059 |
| `rousettus_green_bg_manifest.csv` | `51561ca785d9d3bd` | `6fde60cd4bf9a67a` | 1093 |
| `rousettus_manifest.csv` | `7e51a45d8854e626` | `1b5f989257bcc1f2` | 1059 |
| `rousettus_occ_roi_centre_manifest.csv` | `0e60e3d0a271c0df` | `0881795dc2d9e610` | 1042 |
| `rousettus_occ_roi_matched_manifest.csv` | `c13107252564e8a1` | `1fb9245c5d21ee8c` | 1042 |
| `rousettus_occ_roi_none_manifest.csv` | `8d3dc7ebef7fb6c9` | `979e700ddc1e7c00` | 1042 |
| `rousettus_occ_roi_periphery_manifest.csv` | `a89ab5888c3d8100` | `fc9afc8771420f94` | 1042 |
| `rousettus_original_aligned_224_manifest.csv` | `66f70e2e6e07fcc4` | `b8fb891f21f32972` | 1092 |
| `rousettus_original_aligned_320_manifest.csv` | `2c5532247fae2f0d` | `ec6bc4bd5c68673a` | 1092 |
| `rousettus_original_bg_manifest.csv` | `0faa8d5e3cd6780b` | `1144944d24f59576` | 1059 |
| `rousettus_random_aligned_224_manifest.csv` | `c86c32368d472a52` | `66898f1f1169a4dd` | 1092 |
| `rousettus_random_aligned_320_manifest.csv` | `647e16d4a433152a` | `9dab296932acb7d6` | 1092 |

## Regenerate afterwards (these key on basename)

- `outputs/quality/native_boxes.parquet` — `scripts/measure_native_boxes.py`
- `outputs/quality/battery.parquet` — `scripts/build_quality_battery.py`

## Reproduce

```bash
uv run python scripts/migrate_data_layout.py --species all   # dry run
uv run python scripts/migrate_data_layout.py --species all --execute
```
