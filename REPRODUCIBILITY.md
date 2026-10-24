# Reproducibility instructions

1. Inspect `EXPERIMENT_INDEX.md` and `file_manifest.csv` to locate the original scripts and inputs.
2. Confirm permissions and acquire required NHANES/HRS data from their official providers.
3. Create a clean Python environment; inspect source imports and pin exact package versions.
4. Replace hard-coded local paths **in a separate working copy**, not in archived originals.
5. Follow the original experiment-specific preprocessing and split protocols.
6. Check leakage exclusion and ensure imputation/scaling/OOF fitting use training data only.
7. Run each experiment, record commands, software versions, seeds, data hashes and runtime.
8. Compare regenerated tables/figures/metrics against the manuscript, recording differences.

**Status:** This repository-builder only inventories and documents files. It does not execute
experiments, resolve dependencies, validate reported metrics, or certify external transfer.
`requirements.in` is a candidate list to review, not a verified environment lockfile.
