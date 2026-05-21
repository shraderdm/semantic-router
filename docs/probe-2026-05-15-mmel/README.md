# Python reference probe receipts (Corpus A / #1896 fixtures)

This directory contains the committed measurement artifacts that back the Python-reference cosines cited in upstream PRs against `multi-modal-embed-small`.

## What is measured

Cosine similarity between three Corpus-A fixtures and the 24 candidate anchors from #1896's image-routing pack, computed against five SigLIP-class encoders via HuggingFace `transformers.AutoModel`:

- `google/siglip-base-patch16-512` (the vision tower inside `multi-modal-embed-small`)
- `google/siglip2-base-patch16-naflex`
- `google/siglip2-so400m-patch14-384`
- `llm-semantic-router/multi-modal-embed-large`
- `llm-semantic-router/multi-modal-embed-small`

All five run via the HuggingFace canonical loading path, normalized post-projection, on CPU for cross-model consistency.

## Files

- `candidates_corpus_a.json` - the 24 candidate anchors used (verbatim from `config/signal/embedding/image-routing.yaml` at the time of measurement)
- `cosines_per_candidate_full.csv` - per-encoder x per-fixture x per-candidate cosine matrix; one row per (model, fixture, candidate) tuple. 360 rows: 5 models x 3 fixtures x 24 candidates.

## Example query

To inspect the cosine for `passport_sample.jpg` against `photograph of a passport page` under `multi-modal-embed-small`:

```
awk -F, '$1 == "llm-semantic-router/multi-modal-embed-small" && $3 == "passport_sample.jpg" && $6 == "photograph of a passport page"' cosines_per_candidate_full.csv
```

Returns: `...0.7204`

## How to recompute

Load the same models and candidates via HuggingFace `transformers.AutoModel` and run forward on the corresponding image from `docs/image-fixtures/`. Cosine similarity is computed pairwise between the L2-normalized image embedding and each L2-normalized candidate-text embedding.

Measurement date: 2026-05-15.
