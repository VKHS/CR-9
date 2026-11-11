# Run archive index

Run archives are distributed as release assets
([release/paper_artifacts.json](../release/paper_artifacts.json); the CPU execution in
[release/demo_artifacts.json](../release/demo_artifacts.json)), and
`run_cross_morse.sh reproduce` regenerates the runs ([HPC guide](../docs/HPC.md)).
`index.json` registers archives in the checkout and is empty here.
The index accepts verified archives, one subdirectory per dataset (`arxiv/`,
`proteins/`, `products/`, `roman_empire/` with `split_00` … `split_09`,
`pascalvoc_sp/`, `coco_sp/`, `peptides_func/`), for example the sealed archives
a reproduction packs with `CM_PACK_ARCHIVES=1`. Large archives can be stored in
a persistent deposit; each indexed DOI must identify the actual archive. Git
tracks this guide and `index.json`; [AVAILABILITY.md](../AVAILABILITY.md) states
the release scope.

`index.json` is written exclusively by `scripts/ingest_run.py` (`make ingest-run`), which
verifies the archive's manifest and the sealed stage chain before registering it. The
availability statement and `make paper` read this index; nothing here is edited by hand.

```bash
make ingest-run ARCHIVE=/path/to/arxiv_full.tar.gz
make runs-check          # re-verify every registered archive
make compare-runs        # compare model inventories and observed results with the paper
make paper               # tables, figures and evidence under paper/generated/official
```

When the archive has a real deposit DOI, add `DOI=<actual DOI>` to the ingest
command. A DOI for the software repository is separate from a result-archive DOI.

See [docs/RUN_ARCHIVES.md](../docs/RUN_ARCHIVES.md) for the contract.
