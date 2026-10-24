# Pre-release checklist

- [ ] Verify each file belongs to this manuscript and remove unrelated/obsolete artifacts.
- [ ] Review `file_manifest.csv`, large files, and duplicate outputs.
- [ ] Verify dataset and model redistribution permissions, especially HRS.
- [ ] Remove secrets, credentials, absolute personal paths and sensitive information.
- [ ] Check GitHub file-size limits and externalize large artifacts.
- [ ] Confirm executable entry points and exact package versions.
- [ ] Verify train/test and OOF leakage controls.
- [ ] Re-run experiments and match published results, documenting discrepancies.
- [ ] Choose a code license and check third-party dependencies.
- [ ] Complete CITATION.cff (authors, DOI, release version, date).
- [ ] Create a GitHub release and archive the intended version with Zenodo.
- [ ] Check whether the manuscript's existing Zenodo DOI should receive a new version.
