# Pack-margin calibration receipts (sparse-anchor pack vs 20-image corpus)

This directory contains the committed measurement artifacts that back the pack-margin and per-image cosine numbers cited in upstream PRs from the 2026-05-20 calibration session against `multi-modal-embed-small`.

## What is measured

For a 15-anchor sparse pack (subset of the #1896 anchor list spanning the `identifier_document_imagery`, `code_or_terminal_imagery`, and `ambient_office_imagery` rules), per-image cosine similarity is computed against every candidate anchor in the pack. The 20 images split into three buckets: 8 in-rule positives, 4 adversarial near-misses, and 4 out-of-distribution negatives.

The same measurement is run twice: once with the candle-binding SigLIP vision encoder PRE-normalization (the pre-PR state) and once POST-normalization (the state introduced by the upstream PR). The delta isolates the normalization fix's empirical impact on pack-margin viability.

## Files

- `calibration_2026_05_20_prenorm_full.csv` - per-image cosine against every candidate anchor in the pack, PRE-normalization. One row per image; one column per candidate.
- `calibration_2026_05_20_postnorm_full.csv` - same layout, POST-normalization.
- `calibration_2026_05_20_prenorm_summary.csv` - per-image rollup PRE-normalization: bucket, per-rule max cosine, top rule, top anchor within the top rule, top cosine.
- `calibration_2026_05_20_postnorm_summary.csv` - same layout, POST-normalization.

## Example queries

Top cosine for `inrule_identifier_passport.jpg` POST-normalization:

```
awk -F, '$1 == "inrule_identifier_passport.jpg" {print $8}' calibration_2026_05_20_postnorm_summary.csv
```

Returns: `0.6991`

Top cosine for the same image PRE-normalization:

```
awk -F, '$1 == "inrule_identifier_passport.jpg" {print $8}' calibration_2026_05_20_prenorm_summary.csv
```

Returns: `0.6843`

Margin (in-rule floor minus adversarial ceiling) on `ambient_office_imagery` POST-normalization:

```
awk -F, 'NR>1 && $7 == "ambient_office_imagery" {print $1, $5, $2}' calibration_2026_05_20_postnorm_summary.csv
```

Filter the output by bucket to read in-rule floor (lowest `max_ambient` among `inrule` rows) and adversarial ceiling (highest `max_ambient` among `adversarial` rows).

## Schema

The "full" CSVs use the column layout:

```
image, bucket, <candidate_1>, <candidate_2>, ..., <candidate_N>
```

Where `<candidate_K>` is the short candidate name (for example `passport`, `vscode_code`, `printer`). The text that the encoder actually sees during embedding is the longer descriptive phrase (for example "photograph of a passport page") declared on the corresponding rule in the pack; the short name is the column label only.

The "summary" CSVs use the column layout:

```
image, bucket, max_identifier, max_code, max_ambient, top_rule, top_anchor, top_cosine
```

## How to recompute

Load `multi-modal-embed-small` via HuggingFace `transformers.AutoModel` and run forward on each image in the corpus. Cosine similarity is computed pairwise between the L2-normalized image embedding and each L2-normalized candidate-text embedding. The full pack of 15 anchors is the subset of the #1896 image-routing pack listed in the column headers of the "full" CSVs.

Measurement date: 2026-05-20.
