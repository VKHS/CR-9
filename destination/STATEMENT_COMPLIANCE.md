# Compliance with the data-availability statement — at the final destination

[Now](../STATEMENT_COMPLIANCE.md) · [The final destination](../docs/FINAL_DESTINATION.md) · [destination/](README.md)

> The complete experiment code, configurations, source data, split and seed manifests, environment specification, checkpoints, predictions, propensity-logged trajectories, statistical outputs and figure/table generators are available in the CROSS-MORSE GitHub repository at https://github.com/VKHS/CROSS-MORSE. The repository provides the one-command reconstruction of every reported table and figure and follows established reproducibility guidance.

This page is the projection of `STATEMENT_COMPLIANCE.md` once the measured outputs are published: each item is evaluated on the files of `destination/` with the checks of `scripts/statement_check.py`. Projected states are in $\color{red}{\textsf{red}}$. `make statement-fulfilled` decides the real state after publication.

| Statement item | Now | At the destination | Fulfilled by |
|---|---|---|---|
| complete experiment code | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | E12 adapter code of the fourteen external comparisons at their pinned commits (experiments/baselines/, from records:baselines) |
| configurations | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | E12 study plans: 48 HPO configurations and 12 optimization seeds per comparison |
| source data | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | paper/source_data/ written from the published outputs |
| split and seed manifests | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | manifests/published/: executed memberships of the seven runs (Roman-empire: ten splits) |
| environment specification | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | environment/observed/ of the seven runs and the release asset environment:gpu |
| checkpoints | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | checkpoints in the complete tier of every run |
| predictions | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | prediction arrays in the complete tier of every run |
| propensity-logged trajectories | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | propensity-logged trajectories of every run |
| statistical outputs | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | every run, measurement-record set and campaign published |
| figure/table generators | Fulfilled | Fulfilled | a generator for each numbered display (now) |
| one-command reconstruction of every reported table and figure | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | `make paper`: every display rebuilt and every compared value agrees |
| follows established reproducibility guidance | Fulfilled | Fulfilled | REPRODUCIBILITY.md (now) |

**$\color{red}{\textsf{12 of 12}}$** registered items are fulfilled on the destination tree.

## Every reported table and figure

`make paper` rebuilds 90 numbered displays. 4982 printed values are registered as comparison targets now; at the destination every one of them agrees, and the 11 quantitative figures below carry their own registered contracts as well:

| Figure | Work that closes the contract | Input |
|---|---|---|
| main:Figure 2 | scenario clarification; original plotted data | the reference-ESS scenario and candidate/stage population; raw P coordinates where S14 prints a bound |
| main:Figure 3 | original plotted data | mechanism-contrast families, within-depth action entropies and action-origin fractions of the original records |
| main:Figure 4 | original plotted data; reference work | panel 4b feedback risk and work by ensemble size from the original selection arrays; other panels from S16, S15, S23 and S41 |
| main:Figure 6 | scenario clarification | the declared settings of the decoding-cost scenario of S47 |
| supplementary:Figure S2 | original plotted data | per-instance processed edges, wall time and peak memory of runtime_scaling |
| supplementary:Figure S3 | original plotted data | the complete comparator-grid losses of the ensemble grid |
| supplementary:Figure S4 | original plotted data; scope clarification | scores and provenance of GRAMA/arxiv, AMP/Roman and AMP/Pascal outside the fourteen E12 pairs |
| supplementary:Figure S14 | original plotted data | depth curves, effects by depth, quartiles and phase allocations of depth_diagnostics |
| supplementary:Figure S16 | original plotted data | horizon coverage, NIS coverage and false-safe/harm-miss rates of the reliability records |
| supplementary:Figure S17 | original plotted data; scenario clarification | the held-out critic-RMSE band endpoints |
| supplementary:Figure S18 | scenario clarification; original analytical grid | the nominal baseline and density-scale grid of panel c |

## Scientific consistency

Each open question between the printed methods or values and the code is decided by an original record and a registered rule ([docs/CONSISTENCY.md](../docs/CONSISTENCY.md); `make consistency-audit RUNS_ROOT=<runs>` evaluates the checkable rules). At the destination every item is decided:

| Item | Question | Now | At the destination |
|---|---|---|---|
| K1 | Studentized maximum of the selection bootstrap (Algorithm S4) | code aligned; records decide the printed values | $\color{red}{\textsf{decided by the original records}}$ |
| K2 | Peptides-func confirmation: Holm P = 0.070 above 0.05 | awaiting original records | $\color{red}{\textsf{decided by the original records}}$ |
| K3 | Selection on the complete candidate ledger of Roman-empire, PascalVOC-SP and COCO-SP | awaiting original records | $\color{red}{\textsf{decided by the original records}}$ |
| K4 | Seed hierarchy of Table S2 | design difference; records decide | $\color{red}{\textsf{decided by the original records}}$ |
| K5 | Settings of the budget-matched comparison (Tables S46b and S47, main Figure 6) | needs an author declaration | $\color{red}{\textsf{decided by the author's declaration}}$ |
| K6 | Failure responses of the target-regime gates (Table S3) | awaiting original records | $\color{red}{\textsf{decided by the original records}}$ |
| K7 | Table S20 \|Q\|: the feasible family, not the registry | resolved | resolved |
| K8 | Literal cells of the protocol displays | needs an author declaration | $\color{red}{\textsf{decided by the author's declaration}}$ |
| K9 | Manuscript corrections found while registering the figure contracts | needs an author declaration | $\color{red}{\textsf{decided by the author's declaration}}$ |

## The printed availability text

| Claim | Now | At the destination | Route |
|---|---|---|---|
| M25: the role membership of D_fit, D_explore, D_select, D_confirm, D_compare, D_admit… | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
| M27: are deposited at the Zenodo DOIs listed in the repository’s runs/index.json | Awaiting measured outputs | $\color{red}{\textsf{Route-dependent}}$ | route R: fulfilled with the aligned wording; route Z: fulfilled once the run archives are deposited and registered (section E) |
| M28: the measurement records of the experiments outside the eight-stage run (per-seed… | Missing | $\color{red}{\textsf{Route-dependent}}$ | route R: fulfilled with the aligned wording; route Z: needs record-set deposit registration (section E) |
| M29: make paper rebuilds every run-derived table and figure of this article from the … | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
| M30: make records-tables prints the record-derived supplementary tables, with every v… | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
| R01: The result registry links every table and figure value to its source record, dat… | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
| S03: The run archive of every dataset [...] are deposited at the DOIs listed in the r… | Awaiting measured outputs | $\color{red}{\textsf{Route-dependent}}$ | route R: fulfilled with the aligned wording; route Z: fulfilled once the run archives are deposited and registered (section E) |
| S04: the measurement records of the experiments outside the eight-stage run are depos… | Missing | $\color{red}{\textsf{Route-dependent}}$ | route R: fulfilled with the aligned wording; route Z: needs record-set deposit registration (section E) |
| S05: make paper rebuilds every run-derived table and figure of this Supplementary Inf… | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
| S06: make records-tables prints the record-derived tables, each value traced to the r… | Awaiting measured outputs | $\color{red}{\textsf{Fulfilled}}$ | published with the release paper-artifacts |
