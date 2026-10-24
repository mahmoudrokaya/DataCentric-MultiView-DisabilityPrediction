# Data-Centric Multi-View Disability Risk Prediction — Experimental Evidence

Companion experimental repository for **A Data-Centric Multi-View Learning Framework for Robust Disability Risk Prediction and Cross-Dataset Generalization**.

## Repository organization
This release retains the **original `Collected/` directory layout** rather than imposing new empty directories.
See [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) for a generated directory index and
[structure_tree.txt](structure_tree.txt) for the actual inventory.

## Research workflow
NHANES/HRS data → outcome and leakage audit → dependency graph → constrained views →
view-specific supervised learning → fusion/OOF stacking → validation and regularization.

## Reproducibility
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). File inclusion does **not** establish that
the manuscript's reported metrics have been reproduced.

## Data and redistribution
See [DATA_ACCESS.md](DATA_ACCESS.md) before making this repository public. NHANES and
HRS have separate access conditions. The presence of files does not grant redistribution rights.

## Citation and licensing
`CITATION.cff` is a template: complete author metadata, release version and DOI when known.
Choose a license deliberately; `LICENSE-TO-SELECT.md` is **not** a software license.

## Provenance
`file_manifest.csv` inventories existing experimental files. Run the builder with `--hash`
to generate SHA-256 values. `repository_audit.json` records inventory counts and scan issues.
