"""Build quantitative supplementary displays from verified scientific inputs.

No measured cell is read from the manuscript registry. Record tables retain their
original supplied intervals; joint S14 uses its dedicated validator; budget plots
use the strict campaign replay. Historical reproduction and a complete paper are
separate claims. Every figure has a numerical CSV and every input/output is hashed.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import shutil
import tempfile

from cm_campaigns.common import canonical, confined, directory_inventory, read_json, require, sha256, unique
from cm_campaigns.contract import ADMITTED, ARMS
from scripts.records import ensemble_law, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "cross-morse-supplement-build-1"
FACTORIAL_CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")
ENSEMBLES = (1, 2, 4, 8, 12, 16, 20)
FACTORIAL_DATASETS = {"ogbn-arxiv": ("S16", "arxiv", ("A", "O", "X")),
                      "PascalVOC-SP": ("S16b", "pascal", ("S", "O", "X"))}
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "Roman", "PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}
COLORS = ("#2864a5", "#d37a18", "#467e4c", "#8055a5")


def _number(value, label):
    require(not isinstance(value, bool), f"{label}: a boolean is not a measurement")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: a numeric observation is required") from exc
    require(math.isfinite(number), f"{label}: nonfinite observation")
    return number


def _digest_inventory(inventory):
    return hashlib.sha256(canonical(inventory)).hexdigest()


def _input_identity(path, kind, *, inventory=None, **details):
    path = Path(path)
    files = directory_inventory(path) if inventory is None else inventory
    return {"input_kind": kind, "source": str(path.resolve()), "sha256": _digest_inventory(files),
            "files": files, **details}


def _file_identity(path, kind):
    return {"input_kind": kind, "source": str(Path(path).resolve()), "sha256": sha256(Path(path)),
            "bytes": Path(path).stat().st_size}


def _implementation_identity():
    """Hash local generator/replay code, separately from archived analysis code."""
    paths = set()
    for folder in ("paper/build", "cm_campaigns", "cm_experiments", "cross_morse"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.update(ROOT / name for name in ("scripts/records.py", "scripts/build_supplement.py",
                                         "environment/requirements-cpu.lock"))
    files = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(paths)]
    return {"source": str(ROOT), "sha256": _digest_inventory(files), "files": files,
            "scope": "Builder/replay implementation and pinned dependency contract; archived experiment analysis code remains in each full source inventory"}


def _guard_output(output_dir, inputs):
    raw = Path(output_dir).absolute()
    require(not any(path.is_symlink() for path in (raw, *raw.parents)), "Output path may not contain symlinks")
    output = raw.resolve()
    for source in inputs:
        if source is None:
            continue
        source = Path(source).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "Output directory must be separate from all input evidence directories")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Choose a fresh or empty output directory; existing files are preserved")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _format(value):
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _csv(root, relative, columns, records):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        # CSVs are the exact numeric plot sources; do not truncate binary64
        # inputs to the shorter precision used in the human-readable tables.
        writer.writerows([repr(row[column]) if isinstance(row.get(column), float) else _format(row.get(column))
                          for column in columns] for row in records)
    return relative


def _tex(value):
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c) for c in _format(value))


def _table(root, stem, title, columns, records, note):
    if _BANNER:
        note = f"{_BANNER[0]}. {note}"
    paths = [_csv(root, f"tables/{stem}.csv", columns, records)]
    escape_md = lambda value: _format(value).replace("|", r"\|").replace("\n", " ") or "—"
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(escape_md(row.get(c)) for c in columns) + " |" for row in records]
    lines += ["", note, ""]
    path = root / f"tables/{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    paths.append(path.relative_to(root).as_posix())
    lines = ["% " + title, r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\caption{" + _tex(title) + "}",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
             " & ".join(_tex(c) for c in columns) + r" \\", r"\hline"]
    lines += [" & ".join(_tex(row.get(c)) or "--" for c in columns) + r" \\" for row in records]
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip\parbox{\linewidth}{\scriptsize " + _tex(note) + "}", r"\end{table}", ""]
    path = root / f"tables/{stem}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    paths.append(path.relative_to(root).as_posix())
    return paths


# The CPU demonstration (scripts/reconstruct_paper.py --demo) builds these displays from fixture runs at reduced
# scale; build_supplement(demonstration=True) sets this banner, which every table note and figure written here carries.
DEMONSTRATION_BANNER = "CPU DEMONSTRATION at reduced scale (fixture runs) \u2014 not the paper's measurements"
_BANNER: list[str] = []


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "cross-morse-supplement", "savefig.dpi": 180})
    return plt


def _save(root, figure, stem):
    (root / "figures").mkdir(exist_ok=True)
    if _BANNER:
        figure.text(0.5, 1.0, _BANNER[0], ha="center", va="bottom", color="#b00020", fontsize=8)
    paths = []
    for suffix in ("png", "svg"):
        path = root / f"figures/{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", metadata={"Date": None} if suffix == "svg" else {"Software": None})
        paths.append(path.relative_to(root).as_posix())
    return paths


def _output(identifier, files, displays, coverage, input_kind, note, **extra):
    return {"id": identifier, "files": files, "display_ids": displays, "coverage": coverage,
            "input_kind": input_kind, "synthetic": False, "note": note, **extra}


def _records_index(path, manuscript_root):
    index = read_json(Path(path))
    require(isinstance(index, dict) and index.get("schema") == "cross-morse-records-index-1", "Wrong records index schema")
    entries = index.get("sets")
    require(isinstance(entries, list) and all(isinstance(row, dict) and isinstance(row.get("set"), str) for row in entries),
            "Records index must contain named set entries")
    unique([row["set"] for row in entries], "record set registration")
    specs_path = Path(manuscript_root) / "records/specs.json"
    specs = load_specs(Path(path).parent, specs_path)
    require(all(row["set"] in specs["sets"] for row in entries), "Unrecognized registered record set")
    return {row["set"]: row for row in entries}, specs


def _record_values(index_path, entry, specs, *, demonstration=False):
    """The verified rows of one record set. The CPU demonstration's sets derive from fixture runs and carry their
    synthetic flag; ``demonstration`` admits that flag (the outputs are labelled), never an example set."""
    name = entry["set"]
    path = confined(Path(index_path).parent, f"{name}/MANIFEST.json").parent
    before = directory_inventory(path)
    require(entry.get("verified") is True, f"{name}: record set is not verified in the supplied index")
    checked = verify_set(path, specs["sets"][name], specs)
    require(checked["manifest_sha256"] == entry.get("manifest_sha256"), f"{name}: stale index manifest identity")
    require(demonstration or (checked.get("synthetic") is False and entry.get("synthetic", False) is False),
            f"{name}: synthetic examples cannot be scientific display inputs")
    provenance = read_json(path / "provenance.json")
    require(not provenance.get("example"), f"{name}: example records cannot establish benchmark displays")
    for field in () if demonstration else ("evidence_kind", "data_kind", "status"):
        value = str(provenance.get(field, "")).lower()
        require(not any(word in value for word in ("synthetic", "planning", "advisory", "sensitivity", "template", "example", "toy")),
                f"{name}: {field} classifies this as non-measured evidence")
    rows = read_summary(path, specs["sets"][name], specs["summary_columns"])
    require(directory_inventory(path) == before, f"{name}: source files changed during verification")
    return rows, _input_identity(path, "demonstration_record_set" if demonstration else "verified_measurement_record_set",
                                 inventory=before, record_set=name, manifest_sha256=checked["manifest_sha256"],
                                 **({"synthetic": bool(checked.get("synthetic") or entry.get("synthetic", False))}
                                    if demonstration else {}),
                                 scope="Every summary value traced to a sealed raw file; historical execution and the original inferential method are not independently attested")


def _keyed_values(rows, keys, expected_keys, quantities, label):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        require(key in expected_keys, f"{label}: unexpected row identity {key}")
        values = grouped.setdefault(key, {})
        require(row["quantity"] in quantities and row["quantity"] not in values, f"{label}: duplicate or unexpected quantity")
        require(isinstance(row.get("unit"), str) and bool(row["unit"].strip()), f"{label}: original quantity units must be explicit")
        values[row["quantity"]] = _number(row["value"], f"{label}/{key}/{row['quantity']}")
    require(set(grouped) == set(expected_keys), f"{label}: incomplete registered row coverage")
    require(all(set(values) == set(quantities) for values in grouped.values()), f"{label}: incomplete printed quantities")
    return grouped


def _nats(rows, quantities, label):
    require(all(row["unit"].strip().lower() in ("nat", "nats") for row in rows if row["quantity"] in quantities),
            f"{label}: loss quantities must be recorded in nats")


def _factorial_data(rows, specs):
    expected = {(dataset, cell) for dataset in FACTORIAL_DATASETS for cell in FACTORIAL_CELLS}
    quantities = list(specs["sets"]["factorials"]["quantities"])
    values = _keyed_values(rows, ("dataset", "cell"), expected, quantities, "factorials")
    _nats(rows, ("ld_integrated_loss", "interval_low", "interval_high"), "factorials")
    for dataset in FACTORIAL_DATASETS:
        for quantity in quantities:
            require(len({row["unit"].strip().lower() for row in rows if row["dataset"] == dataset and row["quantity"] == quantity}) == 1,
                    "factorials: unit/denominator changes between cells")
    for row in rows:
        unit = row["unit"].strip().lower()
        if row["quantity"] == "official_metric":
            require(unit in ("%", "percent", "percentage points", "accuracy percent", "f1 points", "percent or f1 points"),
                    "factorials: official metric must be in percent/F1 points, not fractions")
        if row["quantity"] == "activity":
            require(unit in ("fraction", "proportion"), "factorials: activity must be a fraction")
    output = []
    for dataset, (_, _, factors) in FACTORIAL_DATASETS.items():
        for cell in FACTORIAL_CELLS:
            value = values[dataset, cell]
            require(0 <= value["interval_low"] <= value["ld_integrated_loss"] <= value["interval_high"],
                    "factorials: invalid loss interval")
            require(0 <= value["official_metric"] <= 100 and 0 <= value["activity"] <= 1 and value["relative_time"] > 0,
                    "factorials: metric/activity/time outside their domains")
            time_unit = next(row["unit"] for row in rows if row["dataset"] == dataset and row["cell"] == cell and row["quantity"] == "relative_time")
            output.append({"dataset": dataset, "cell": cell, **dict(zip(factors, map(int, cell))), **value,
                           "relative_time_unit": time_unit})
    return output


def _factorial_displays(root, records):
    plt = _plt()
    outputs, contrasts = [], []
    for dataset, (table, slug, factors) in FACTORIAL_DATASETS.items():
        data = [row for row in records if row["dataset"] == dataset]
        columns = ["cell", *factors, "ld_integrated_loss", "interval_low", "interval_high", "official_metric", "activity", "relative_time", "relative_time_unit"]
        note = ("All eight cells and their six measured quantities are read from verified factorial records. Loss and original 95% interval are in nats; official metric is in percent/F1 points, activity is a fraction. "
                "Relative time retains the explicitly recorded unit/denominator without renormalization. Cell intervals are preserved; contrast intervals need the original joint resampling covariance and are not inferred from marginal intervals.")
        paths = _table(root, table, f"{table}: {dataset} factorial cell measurements", columns, data, note)
        outputs.append(_output(f"factorial_{slug}_table", paths, [f"supplementary:Table {table}"], "columns",
                               "verified_measurement_record_set", note, complete_cell_rows=True, row_count=8))
        numeric = _csv(root, f"data/factorial_{slug}.csv", ["dataset", *columns], data)
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        for index, row in enumerate(data):
            ax.errorbar(index, row["ld_integrated_loss"],
                        yerr=[[row["ld_integrated_loss"] - row["interval_low"]], [row["interval_high"] - row["ld_integrated_loss"]]],
                        fmt="o", capsize=3, color=COLORS[0] if row["cell"] != "111" else COLORS[1])
        ax.set_xticks(range(8), [row["cell"] for row in data])
        ax.set_xlabel("Cell (" + ", ".join(factors) + ")")
        ax.set_ylabel("Log-depth-integrated CE (nats)")
        ax.set_title(f"{dataset}: equal-budget factorial")
        ax.grid(axis="y", alpha=.2)
        fig.text(.5, .005, "Original cell 95% intervals; no independent-cell assumption", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, 1))
        paths = [numeric, *_save(root, fig, f"factorial_{slug}")]
        plt.close(fig)
        displays = ["main:Figure 4"] if dataset == "ogbn-arxiv" else ["supplementary:Table S16b"]
        outputs.append(_output(f"factorial_{slug}_plot", paths, displays,
                               "partial_panels" if dataset == "ogbn-arxiv" else "columns", "verified_measurement_record_set",
                               "Cell-mean/interval plot: main Figure 4a for arxiv; companion view of S16b for Pascal. Other panels and joint-covariance contrast intervals are not supplied."))
        means = {row["cell"]: row["ld_integrated_loss"] for row in data}
        for mask in range(1, 8):
            selected = [index for index in range(3) if mask & (1 << index)]
            name = ":".join(factors[index] for index in selected)
            effect = math.fsum(math.prod(1 if int(cell[index]) else -1 for index in selected) * value
                               for cell, value in means.items()) / 4
            anchored = math.fsum(((-1) ** (len(selected) - sum(int(cell[index]) for index in selected))) * value
                                 for cell, value in means.items() if all(int(cell[index]) == 0 for index in range(3) if index not in selected))
            contrasts.append({"dataset": dataset, "contrast": name, "effect_coded_high_minus_low_nats": effect,
                              "baseline_anchored_difference_nats": anchored, "interval": None})
    paths = _table(root, "factorial_point_contrasts", "Factorial point contrasts from the eight paired cell means",
                   ["dataset", "contrast", "effect_coded_high_minus_low_nats", "baseline_anchored_difference_nats", "interval"], contrasts,
                   "Effect coding averages the signed cells with divisor four. Baseline-anchored finite differences hold other factors at zero. These conventions differ for interactions. No contrast interval is produced without the original joint resampling covariance.")
    outputs.append(_output("factorial_point_contrasts", paths, ["supplementary:Table S16", "supplementary:Table S16b"],
                           "columns", "derived_from_verified_measurement_records", "Point contrasts only; no fabricated interaction confidence intervals."))
    return outputs


def _ensemble_data(rows, specs):
    expected = {(dataset, str(ensemble)) for dataset in ADMITTED for ensemble in ENSEMBLES}
    values = _keyed_values(rows, ("dataset", "ensemble"), expected, ("loss_nats",), "ensemble_grid")
    _nats(rows, ("loss_nats",), "ensemble_grid")
    fits, points = [], []
    for dataset in ADMITTED:
        observed = [(ensemble, values[dataset, str(ensemble)]["loss_nats"]) for ensemble in ENSEMBLES]
        require(all(loss >= 0 for _, loss in observed), "ensemble_grid: loss must be nonnegative")
        fit = ensemble_law(observed)
        fits.append({"dataset": dataset, **fit, "ensemble_sizes": ",".join(map(str, ENSEMBLES)),
                     "positive_D_inf": fit["D_inf"] > 0, "nonnegative_asymptote": fit["l_inf"] >= 0})
        for ensemble, loss in observed:
            fitted = fit["l_inf"] + fit["D_inf"] / ensemble
            points.append({"dataset": dataset, "ensemble": ensemble, "inverse_ensemble": 1 / ensemble,
                           "loss_nats": loss, "fitted_loss_nats": fitted, "residual_nats": loss - fitted})
    return fits, points


def _ensemble_displays(root, data):
    fits, points = data
    note = ("OLS of the verified comparator losses on 1/E over the exact E={1,2,4,8,12,16,20} grid. The model's coefficient/asymptote diagnostics are reported even when unfavorable. "
            "The original record schema does not supply the printed D_inf band, plant-loss ratio kappa, break-even cost inputs, feedback-by-E response or extra-checkpoint accounting; none is inferred from the manuscript.")
    tables = _table(root, "S26b_fit", "S26b: comparator ensemble-law fit and residual diagnostics",
                    ["dataset", "ensemble_sizes", "l_inf", "D_inf", "max_abs_residual", "positive_D_inf", "nonnegative_asymptote"], fits, note)
    outputs = [_output("ensemble_law_fit", tables, ["supplementary:Table S26b"], "columns", "derived_from_verified_measurement_records", note)]
    numeric = _csv(root, "data/ensemble_grid.csv", ["dataset", "ensemble", "inverse_ensemble", "loss_nats", "fitted_loss_nats", "residual_nats"], points)
    plt = _plt()
    paths = [numeric]
    for residuals in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0))
        for ax, dataset in zip(axes.flat, ADMITTED):
            rows = [row for row in points if row["dataset"] == dataset]
            x = [row["ensemble"] for row in rows]
            if residuals:
                ax.axhline(0, color="0.4", lw=.8)
                ax.plot(x, [row["residual_nats"] for row in rows], "o-", color=COLORS[1], ms=4)
                ax.set_ylabel("Observed minus fitted (nats)")
            else:
                ax.plot(x, [row["loss_nats"] for row in rows], "o", label="Original grid", color=COLORS[0], ms=4)
                fit = next(row for row in fits if row["dataset"] == dataset)
                grid = [1 + i * 19 / 200 for i in range(201)]
                ax.plot(grid, [fit["l_inf"] + fit["D_inf"] / n for n in grid], "-", label="OLS: l_inf + D_inf/E", color=COLORS[1])
                ax.set_ylabel("Comparator loss (nats)")
            ax.set_xticks((1, 4, 8, 12, 16, 20))
            ax.set_xlabel("Ensemble size E")
            ax.set_title(SHORT[dataset])
            ax.grid(axis="y", alpha=.2)
        if not residuals:
            axes[0, 0].legend(frameon=False)
        fig.suptitle("Comparator ensemble-law residuals" if residuals else "Original comparator ensemble grid and fitted law")
        fig.tight_layout(rect=(0, 0, 1, .95))
        paths += _save(root, fig, "ensemble_residuals" if residuals else "ensemble_fit")
        plt.close(fig)
    outputs.append(_output("ensemble_grid_diagnostics", paths, ["supplementary:Table S26b"], "columns",
                           "derived_from_verified_measurement_records", "Measured comparator points and fit diagnostics. These do not reconstruct Figure 4b's feedback effect/work response or the S3 ensemble sensitivity band."))
    return outputs


def _confirmation_data(campaign, manuscript_root):
    """The validated joint S14 campaign.  check_campaign refuses an invalid campaign; a valid one feeds the
    displays whether or not its values agree with the printed S14 (status PASS or REFERENCE_MISMATCH): that
    agreement is the separate comparison step's verdict (manuscript_comparisons), never a generator input."""
    from cm_campaigns.confirmation import check_campaign
    before = directory_inventory(Path(campaign))
    checked = check_campaign(Path(campaign), manuscript_root=Path(manuscript_root))
    require(checked["status"] in ("PASS", "REFERENCE_MISMATCH"), f"S14 campaign validation returned {checked['status']}")
    inventory = directory_inventory(Path(campaign))
    require(inventory == before and _digest_inventory(before) == checked["input_identity"],
            "S14 input files changed during verification or the build")
    source = _input_identity(campaign, "original_confirmation_campaign", inventory=inventory,
                             original_test_p_values_recomputed=False, verifier_schema=checked["schema"],
                             reference_status=checked["status"], production_kind=checked.get("production_kind"),
                             post_hoc=checked.get("post_hoc", False),
                             scope=checked["scope"], original_run_linkage="Scientific audit separately requires all seven official archives",
                             verification_contract_files=[_file_identity(Path(manuscript_root) / relative, "registered_manuscript_contract")
                                                          for relative in ("paper/sources.json", "paper/registered_constants.json", "paper/reported_results.json")])
    return checked, source


def _confirmation_displays(root, checked):
    rows = [{"contrast": item["contrast_id"], "dataset": item["dataset"], "effect_nats": item["effect_nats"],
             "interval_low_nats": item["interval_nats"][0], "interval_high_nats": item["interval_nats"][1],
             "statistical_unit": item["statistical_unit"], "unit_count": item["unit_count"], "seed_count": item["seed_count"],
             "raw_P_zero": item["p_value_zero"], "joint_Holm_P_zero": item["holm_p_value_zero"],
             "signed_point_margin_nats": item["beneficial_direction"] * item["practical_margin_nats"],
             "retained": item["retained"], "tested_policy_selected": item["tested_policy_selected"],
             "loss": item["loss"], "method": item["method_name"]} for item in checked["rows"]]
    note = ("Nine original tests against zero in one Holm family; retain only P<0.05 and a point effect strictly beyond the signed practical margin. "
            "Original raw P values and 95% intervals are preserved, paired effects and pooled Holm are replayed. Tests against the practical margin, equivalence tests and S14b rescaling are separate and not reconstructed here. "
            "Unselected fixed-family feedback contrasts remain in the family without changing a selected comparator. Official-run linkage requires the separate scientific audit.")
    columns = list(rows[0])
    paths = _table(root, "S14_joint", "S14: original joint nine-test confirmation family", columns, rows, note)
    (root / "data").mkdir(exist_ok=True)
    (root / "data/S14_joint_audit.json").write_bytes(canonical(checked))
    paths.append("data/S14_joint_audit.json")
    outputs = [_output("joint_s14_confirmation_table", paths, ["supplementary:Table S14"], "columns", "original_confirmation_campaign", note,
                       original_test_p_values_recomputed=False)]
    numeric = _csv(root, "data/S14_joint_forest.csv", columns, rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.1, 5.5))
    for index, row in enumerate(rows):
        color = COLORS[0] if row["retained"] else "#9a4b32"
        ax.plot([row["interval_low_nats"], row["interval_high_nats"]], [index, index], color=color, lw=1.7)
        ax.plot(row["effect_nats"], index, "o", color=color, mfc=color if row["retained"] else "white", ms=5)
        ax.plot(row["signed_point_margin_nats"], index, "|", color=COLORS[2], ms=13, mew=1.6)
        ax.text(1.02, index, f"Holm P={row['joint_Holm_P_zero']:.4g}; {'retain' if row['retained'] else 'do not retain'}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="0.5", lw=.8)
    unit_labels = {"buffered_cluster": "buffered clusters", "official_split": "official splits", "graph": "graphs"}
    ax.set_yticks(range(len(rows)), [f"{row['contrast']} ({row['unit_count']} {unit_labels[row['statistical_unit']]})" for row in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Signed paired loss contrast (nats); tick: practical point margin")
    ax.set_title("Original joint S14 family: effects and 95% intervals")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .01, "Original heterogeneous tests preserved; raw P values are not recomputed. No S14b sample-size rescaling.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.27, right=.7, bottom=.13, top=.9)
    paths = [numeric, *_save(root, fig, "S14_joint_forest")]
    plt.close(fig)
    outputs.append(_output("joint_s14_confirmation_forest", paths, ["main:Figure 3", "supplementary:Figure S9"], "partial_panels",
                           "original_confirmation_campaign", "State/RH contrasts support Figure 3a-b; action entropy/origins in 3c-d are absent. Related S9 contrast view only: the J_old/J_new sensitivity is not supplied.",
                           original_test_p_values_recomputed=False))
    return outputs


def _budget_data(campaign, *, demonstration=False):
    """The strictly replayed campaign. The CPU demonstration's campaign runs on fixture parents without scheduler
    accounting, which the strict check refuses; ``demonstration`` replays it without those qualifications and
    still requires all 24 contrasts of the registered family."""
    from cm_campaigns.arm_campaign import check_campaign
    path = Path(campaign)
    before = directory_inventory(path)
    checked = check_campaign(path, read_json(confined(path, "study.json")), strict=not demonstration)
    require(checked.get("passed") is True and len(checked.get("summary", [])) == 24,
            "Budget displays require 24 replayed contrasts" if demonstration else "Budget displays require 24 strictly replayed contrasts")
    require(directory_inventory(path) == before, "Budget campaign files changed during verification")
    return checked, _input_identity(path, "demonstration_budget_campaign" if demonstration else "strict_budget_campaign",
                                   inventory=before, verifier_scope=checked.get("scope"),
                                   historical_table_reproduction=checked.get("historical_table_reproduction", False))


def _budget_displays(root, checked):
    observed = {(row["dataset"], row["arm"]): row for row in checked["summary"]}
    require(len(observed) == 24 and set(observed) == {(dataset, arm) for dataset in ADMITTED for arm in ARMS}, "Budget contrast coverage differs from the registered four-arm family")
    rows = []
    for dataset in ADMITTED:
        for arm in ARMS:
            item = observed[dataset, arm]
            row = {"dataset": dataset, "arm": arm, "effect_vs_qC_nats": _number(item["effect_vs_qC"], "qC effect"),
                   "effect_vs_qM_nats": _number(item["effect"], "qM effect"), "ci_low_vs_qM_nats": _number(item["ci_low"], "interval"),
                   "ci_high_vs_qM_nats": _number(item["ci_high"], "interval"), "joint_Holm_P": _number(item["p_holm"], "P"),
                   "work_ratio_qC": _number(item["work_ratio"], "work"), "ceiling_ratio_qC": _number(item["ceiling"], "ceiling"),
                   "teacher_work_ratio_qC": _number(item["reference_work_ratio"], "teacher work"),
                   "seeds": item["seeds"], "units": item["units"], "inference_scope": item["inference_scope"]}
            require(row["ceiling_ratio_qC"] > 0 and row["work_ratio_qC"] >= 0 and 0 <= row["joint_Holm_P"] <= 1,
                    "Budget work/ceiling/P values outside their domains")
            require(row["ci_low_vs_qM_nats"] <= row["effect_vs_qM_nats"] <= row["ci_high_vs_qM_nats"], "Budget paired interval does not contain its effect")
            row["work_fraction_of_ceiling"] = row["work_ratio_qC"] / row["ceiling_ratio_qC"]
            row["teacher_effect_vs_qC_nats"] = row["effect_vs_qC_nats"] - row["effect_vs_qM_nats"]
            rows.append(row)
        selected = rows[-4:]
        for key in ("teacher_effect_vs_qC_nats", "teacher_work_ratio_qC", "ceiling_ratio_qC"):
            require(all(math.isclose(row[key], selected[0][key], rel_tol=1e-8, abs_tol=1e-10) for row in selected),
                    "Budget arms do not share one comparator/teacher/ceiling")
    note = ("Twenty-four D_compare contrasts from strict original-file replay. Positive arm-minus-qM effects favor the frozen teacher. Holm operates within each dataset's four-arm family. "
            "Work/ceiling is derived from executed work, not confused with work/qC. Controller seeds remain conditional on the frozen parent ensembles. "
            "This is a new experiment, not historical Table S46 reconstruction; retained-share grid ranges, the teacher's D_test interval and modelled tails are not supplied.")
    paths = _table(root, "S46b_comparison", "S46b: replayed D_compare arm outcomes", list(rows[0]), rows, note)
    (root / "data").mkdir(exist_ok=True)
    # Keep the aggregate scientific result, not the potentially huge per-input reserve listing.
    summary = {key: value for key, value in checked.items() if key != "reserve_check"}
    summary["reserve_trace_passed"] = checked.get("reserve_check", {}).get("reserve_trace_passed")
    (root / "data/budget_audit_summary.json").write_bytes(canonical(summary))
    paths.append("data/budget_audit_summary.json")
    outputs = [_output("budget_comparison_table", paths, ["supplementary:Table S46b"], "columns", "strict_budget_campaign", note,
                       historical_table_reproduction=False)]
    numeric = _csv(root, "data/budget_comparison.csv", list(rows[0]), rows)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 9.0), sharey=True)
    for index, row in enumerate(rows):
        color = COLORS[ARMS.index(row["arm"])]
        axes[0].plot([1000 * row["ci_low_vs_qM_nats"], 1000 * row["ci_high_vs_qM_nats"]], [index, index], color=color, lw=1.4)
        axes[0].plot(1000 * row["effect_vs_qM_nats"], index, "o", color=color, ms=4)
        axes[1].barh(index, row["work_fraction_of_ceiling"], color=color, height=.62)
        axes[1].text(row["work_fraction_of_ceiling"] + .015, index, f"P={row['joint_Holm_P']:.3g}", va="center", fontsize=7)
    axes[0].axvline(0, color="0.4", lw=.8)
    axes[1].axvline(1, color="0.3", ls="--", lw=1)
    axes[0].set_yticks(range(24), [f"{SHORT[row['dataset']]} / {row['arm']}" for row in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Arm minus frozen teacher (millinats)\n95% crossed seed-by-unit interval")
    axes[1].set_xlabel("Executed work / common ceiling\nP: within-dataset four-test Holm value")
    axes[1].set_xlim(0, max(row["work_fraction_of_ceiling"] for row in rows) + .32)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        for separator in (3.5, 7.5, 11.5, 15.5, 19.5):
            ax.axhline(separator, color=".85", lw=.6)
    fig.suptitle("Replayed D_compare arms: paired effects and executed work")
    fig.text(.5, .008, "Conditional on frozen parents; arxiv/proteins/Roman are descriptive dependent-unit comparisons. New experiment, not historical S46.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .96))
    paths = [numeric, *_save(root, fig, "budget_comparison")]
    plt.close(fig)
    outputs.append(_output("budget_comparison_forest_work", paths, ["supplementary:Table S46b"], "columns", "strict_budget_campaign", note,
                           historical_table_reproduction=False))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    frontier = []
    for ax, dataset in zip(axes.flat, ADMITTED):
        selected = [row for row in rows if row["dataset"] == dataset]
        reference = selected[0]
        data = [{"dataset": dataset, "policy": "qC", "work_ratio_qC": 1., "effect_vs_qC_nats": 0.},
                {"dataset": dataset, "policy": "qM", "work_ratio_qC": reference["teacher_work_ratio_qC"],
                 "effect_vs_qC_nats": reference["teacher_effect_vs_qC_nats"]}]
        data += [{"dataset": dataset, "policy": row["arm"], "work_ratio_qC": row["work_ratio_qC"], "effect_vs_qC_nats": row["effect_vs_qC_nats"]} for row in selected]
        for row in data:
            ax.plot(row["work_ratio_qC"], row["effect_vs_qC_nats"] * 1000, "D" if row["policy"] == "qC" else "o",
                    ms=5, label=row["policy"], mfc="white" if row["policy"] in ARMS else None)
            frontier.append(row)
        ax.axvline(reference["ceiling_ratio_qC"], color=".5", ls="--", lw=.8)
        ax.axhline(0, color=".5", lw=.6)
        ax.set_title(SHORT[dataset])
        ax.set_xlabel("Executed work / qC")
        ax.set_ylabel("D_compare effect vs qC (millinats)")
        ax.grid(alpha=.15)
    axes[0, 0].legend(ncol=3, fontsize=7, frameon=False)
    fig.suptitle("D_compare budget frontiers from executed inputs")
    fig.text(.5, .008, "Points only: no D_test teacher interval, configuration-range uncertainty or ensemble sensitivity band is inferred", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    paths = [_csv(root, "data/budget_frontier.csv", ["dataset", "policy", "work_ratio_qC", "effect_vs_qC_nats"], frontier),
             *_save(root, fig, "budget_frontier")]
    plt.close(fig)
    outputs.append(_output("budget_comparison_frontier", paths, ["main:Figure 6", "supplementary:Figure S3"], "partial_panels",
                           "strict_budget_campaign", "Related Figure 6a/S3 executed-work point frontiers on D_compare. Does not supply the printed D_test intervals, configuration ranges, ensemble band, power or tail panels.", historical_table_reproduction=False))
    return outputs


def build_supplement(output_dir, *, records_index=None, confirmation_campaign=None, budget_campaign=None, manuscript_root=ROOT,
                     demonstration=False):
    """Return a hashed build manifest; malformed supplied inputs appear in errors.

    Missing optional sources/registered sets are skipped. A complete output table
    is never equated with historical execution, a full figure, or all 90 displays.
    ``demonstration`` builds the CPU demonstration's displays from its fixture-derived inputs: every output is
    marked synthetic and carries DEMONSTRATION_BANNER; the paper's build never sets it.
    """
    _BANNER[:] = [DEMONSTRATION_BANNER] if demonstration else []
    try:
        return _build_supplement(output_dir, records_index=records_index, confirmation_campaign=confirmation_campaign,
                                 budget_campaign=budget_campaign, manuscript_root=manuscript_root,
                                 demonstration=demonstration)
    finally:
        _BANNER[:] = []


def _build_supplement(output_dir, *, records_index, confirmation_campaign, budget_campaign, manuscript_root, demonstration):
    input_roots = [Path(records_index).parent if records_index is not None else None, confirmation_campaign, budget_campaign]
    out = _guard_output(output_dir, input_roots)
    report = {"schema": SCHEMA, "status": "PENDING", "synthetic": False, "synthetic_examples": False,
              "outputs": [], "inputs": [], "skipped": [], "errors": [],
              "implementation": _implementation_identity(),
              "scope": "Numerical displays from supplied verified records/campaigns; no new benchmark observations and no all-paper reproduction claim"}
    if demonstration:
        report.update(demonstration=True, scope="CPU demonstration at reduced scale: displays from its fixture-derived "
                                                "records and campaigns; not the paper's measurements")
    available, specs = {}, None
    if records_index is not None:
        try:
            available, specs = _records_index(records_index, manuscript_root)
            report["inputs"] += [_file_identity(records_index, "records_index"),
                                  _file_identity(Path(manuscript_root) / "records/specs.json", "registered_record_contract")]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append({"input": "records_index", "error": str(exc)})
    def build(identifier, load, render):
        try:
            value, identity = load()
            with tempfile.TemporaryDirectory(prefix="cm-supplement-", dir=out.parent) as temporary:
                staged = Path(temporary)
                outputs = render(staged, value)
                all_paths = sorted({relative for item in outputs for relative in item["files"]})
                for relative in all_paths:
                    source = confined(staged, relative)
                    destination = out / relative
                    require(not destination.exists(), "Builder output collision")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(source, destination)
                for item in outputs:
                    item["input_sha256"] = identity["sha256"]
                    item["sha256"] = {relative: sha256(out / relative) for relative in item["files"]}
                    if demonstration:
                        # the reconstruction's kind for fixture-derived inputs (scripts/reconstruct.py), which never
                        # counts as a full numerical display
                        item.update(synthetic=True, demonstration=True, source_input_kind=item["input_kind"],
                                    input_kind="synthetic_examples", note=f"{DEMONSTRATION_BANNER}. {item['note']}")
                if demonstration and outputs:
                    report["synthetic"] = report["synthetic_examples"] = True
                report["inputs"].append(identity)
                report["outputs"].extend(outputs)
        except ImportError as exc:
            # the campaign replays verify the parents' checkpoints with the PyTorch runtime
            report["errors"].append({"input": identifier, "error": f"the replay needs the recorded CPU environment ({exc})"})
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            report["errors"].append({"input": identifier, "error": str(exc)})
    for name, prepare, render in (("factorials", _factorial_data, _factorial_displays), ("ensemble_grid", _ensemble_data, _ensemble_displays)):
        if name not in available:
            report["skipped"].append({"input": name, "reason": "No registered measurement record set supplied"})
            continue
        def load_record(name=name, prepare=prepare):
            rows, identity = _record_values(records_index, available[name], specs, demonstration=demonstration)
            return prepare(rows, specs), identity
        build(name, load_record, render)
    for name, source, load, render in (("confirmation_campaign", confirmation_campaign,
                                       lambda: _confirmation_data(confirmation_campaign, manuscript_root), _confirmation_displays),
                                      ("budget_campaign", budget_campaign,
                                       lambda: _budget_data(budget_campaign, demonstration=demonstration), _budget_displays)):
        if source is None:
            report["skipped"].append({"input": name, "reason": "Original campaign not supplied"})
        else:
            build(name, load, render)
    report["status"] = "FAIL" if report["errors"] else "PASS" if report["outputs"] else "PENDING"
    (out / "BUILD.json").write_bytes(canonical(report))
    return report
