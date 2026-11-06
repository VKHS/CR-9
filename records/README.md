# Measurement records outside the eight-stage run

[Repository home](../README.md) · [Run archives](../docs/RUN_ARCHIVES.md) · [Availability](../AVAILABILITY.md) ·
[Implementation notes](../paper/IMPLEMENTATION_NOTES.md)

The eight-stage runner supplies the run-derived columns of main Tables 1–2 and
Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41 through
[run archives](../docs/RUN_ARCHIVES.md). External baselines, input-gate timings,
sensor traces and campaign accounting enter through the measurement sets below.
Each set carries the raw logs its summary values are read from.
`scripts/records.py` traces every summary value to its raw file and builds the
corresponding record table. [AVAILABILITY.md](../AVAILABILITY.md) states the
supported reconstruction scope and current coverage.

## Record sets

Defined in [`specs.json`](specs.json); the directory name of a set is its key.

| Set | Manuscript tables | Experiments | Keys | Quantities |
|---|---|---|---|---|
| `baselines` | S15, S15b | E12, E17 | dataset, method, seed | serial cost, controller wall time, GPU-hours, per-seed official score, HPO trials, parameters |
| `factorials` | S16, S16b | — | dataset, cell | log-depth-integrated loss with its 95% interval, official metric, activity, relative time |
| `negative_controls` | S19 | — | control | residual activity, input fallback, worst pre-fallback harm |
| `input_gate` | S23 | — | dataset | input-fallback probability of `g_input`, deployed wall-time overhead, peak GPU and host memory |
| `nominal_time` | S23b | E17 | dataset | per-input nominal execution time `t_nom` |
| `sensor_parameters` | S6c | E17 | dataset | executed sketch width `k`, sensing period `P_s`, residual fraction `f_res` |
| `cross_backbone` | S25 | — | regime, model | paired effect, outer folds selecting the route, outer folds evaluated |
| `ensemble_grid` | S26b | E14 | dataset, ensemble | comparator loss per ensemble size (the table fits `l(E) = l_inf + D_inf / E`) |
| `inserted_site` | S45 | E13 | dataset, arm | paired effect relative to `q_C`, graph-unit standard error |
| `budget_matched` | S46, S46b | E3–E6 | dataset, arm, configuration | paired effects against `q_C` and `q_M` with 95% interval, Holm-adjusted P, work ratio, ceiling |

## Layout of one set

```text
records/<set>/
├── summary.csv        one measured value per row
├── raw/               the logs the values are read from (training logs, scheduler accounting, timing traces, metric JSON)
├── code/              the scripts, configurations and commit identities that produced them
├── provenance.json    experiment, produced_by, hardware, period, protocol
└── MANIFEST.json      SHA-256 of every file (written by `make records-index`)
```

`summary.csv` has the set's key columns followed by
`quantity,value,unit,source_file,locator`. The shape of three rows of the `baselines` set (angle
brackets mark what the deposited row contains):

```text
dataset,method,seed,quantity,value,unit,source_file,locator
<dataset>,<method>,,gpu_hours,<value>,GPU-h,raw/<scheduler export>.csv,csv:JobName=<job>;GPUHours
<dataset>,<method>,<seed>,score,<value>,%,raw/<metrics>.json,json:/test/score
<dataset>,<method>,,wall_time,<value>,nominal,raw/<timing trace>.log,regex:wall_time_nominal=([0-9.]+)
```

A locator says where in `source_file` the value is:

| Locator | Meaning |
|---|---|
| `json:/a/b/0` | RFC 6901 pointer into a JSON file |
| `csv:column=value;target` | the first row whose `column` equals `value`; the cell in column `target` |
| `regex:pattern` | the first match of a regular expression with one capture group in a text file |

The value in `summary.csv` must equal the raw value at the precision it is printed with: a raw
`4.663` may be printed as `4.66`, not as `4.70` or `4.6600`.

## Writing a set from the experiment outputs

`scripts/export_records.py` writes a set from the outputs the experiment code of this repository produces, so that
no value is typed by hand:

| Set | Command | Reads |
|---|---|---|
| `budget_matched` | `export_records.py budget-matched --campaign <export>` | `analysis.json` of a complete campaign export (`make arm-export`) |
| `baselines` | `export_records.py baselines --study ... --timing ... --accounting ...` | pinned-upstream studies (`scripts/run_upstream_baseline.py`), E17 timing rows, original sacct exports |
| `sensor_parameters` | `export_records.py from-runs --set sensor_parameters --run ...` | each sealed run's `config.json` and the deployed rule's executed actions |
| `nominal_time` | `export_records.py from-runs --set nominal_time --run ...` | each sealed run's nominal-member execution record |

Original outputs are copied into `raw/` unchanged. A computed quantity (a score from saved predictions,
GPU-hours from the scheduler export, a ratio from timing rows, a resolvent share from action counts) is
written into a small JSON file under `raw/` with the SHA-256 of its inputs. Small inputs are copied next to it;
bulky ones (per-fit test predictions, per-unit campaign observations) stay in the archived studies or campaign,
whose DOI and checksum enter `provenance.json`, and are named by SHA-256 and archive path
(`--copy-predictions`, `--copy-observations` copy them when they are small enough for the repository).
`code/SOURCE.json` records the command, the runtime source identity and the repository commit. The
exporter stops instead of filling a gap: a registered seed without its final fit, an allocation missing
from the sacct export, a timing row outside a completed GPU allocation or an incomplete campaign
ends the export. The other six sets are deposited by hand as below, with the producing code under `code/`.

## Synthetic example sets

`examples/records/` holds `sensor_parameters` and `nominal_time` written by the exporter from the two
bundled synthetic example runs. Their `provenance.json` carries `"synthetic": true`, their tables are titled
*SYNTHETIC EXAMPLE* (`paper/generated/examples/records/`), and `make example-records-check` re-traces them in
CI. `scripts/records.py` refuses a synthetic set under `records/`, and the exporter refuses to write one there.

## Depositing a set

```bash
mkdir -p records/baselines/raw records/baselines/code
cp <logs, scheduler exports, metric files> records/baselines/raw/
cp <launch scripts, configurations, commit list> records/baselines/code/
$EDITOR records/baselines/summary.csv records/baselines/provenance.json
make records-index        # hashes the set, traces every value, writes records/index.json and AVAILABILITY.md
make records-check        # what CI runs on every push
make records-tables       # paper/generated/official/records/<set>.{tex,md,csv}
```

Remove allocation identifiers and personal paths from logs before depositing; the release audit
rejects NAISS project identifiers anywhere in the tree.

## What the check establishes

| Check | Establishes | Outside its scope |
|---|---|---|
| `MANIFEST.json` | the deposited files are the files that were indexed | when or where they were produced |
| locator trace | every printed value appears in a deposited raw file | that the raw file comes from the stated experiment; `provenance.json` and `code/` document that |
| printed table | the supplementary table is a function of the deposited summary | statistical claims beyond the summarised values |
