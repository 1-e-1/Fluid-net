# Publication notes

Export date: 2026-10-24. Account: 1-e-1. License selection: MIT for original material.

This repository is an export of an existing local research project. Source files
were copied without moving or deleting the working project. Data archives, model
weights, caches, temporary files, private workflow logs and Word drafts are excluded.
Third-party source licenses and provenance are retained.

Small code checks validate only the exercised interfaces and numerical behavior;
they do not reproduce the historical research results or prove reported advantages.
The existing source is research code and includes historical experimental variants.
No training, data download or paid service is launched by this export process.

Export-only maintenance fix: `improvements/benchmark_all_v2.py` had an existing
validation-loop indentation error and reused stale training logits. The public
copy now indents both branches correctly and extracts validation logits using
the same rule as its training loop. No benchmark results were regenerated.
