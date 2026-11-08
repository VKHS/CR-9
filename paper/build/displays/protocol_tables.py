"""Protocol and definition displays rendered from the repository's configuration and code.

Each generator rebuilds one printed protocol table, or the lifecycle illustration of
main Figure 1, from what the released implementation executes: the full configurations
(``configs/<dataset>_full.json``), ``paper/registered_constants.json``,
``data/sources.json``, ``records/specs.json``, the role contract of
``cross_morse/split_contract.py`` and the registries and literals of the runtime
modules. Runtime modules that import PyTorch (``core``, ``learning``, ``policies``,
``pipeline``, ``training``) are read as source text with :mod:`ast`, never imported.
Prose that no configuration or code expresses is kept in
``paper/protocol/<stem>.json`` with its page provenance.

Tables S5 and S5d also read sealed lifecycle runs (checkpoint inventories, per-seed
training records, identity audits, sealed sensor settings); they are measurement displays.

Printed values are reached only through ``out.compare``. The independently reviewed
``paper/printed_display_values.json`` supplies the protocol comparisons and pins
them to the supplied PDF bytes. It is never read as a computation input.
"""
from __future__ import annotations

import ast
import functools
import json
import math
import re
import statistics
import textwrap
from pathlib import Path

from .framework import (DATASETS, MISSING_TEXT, SHORT, STEMS, DisplayInputError, display, integer, num,
                        not_applicable, pyplot)

PROFILE = "full"
EXAMPLE_RUNS = ("examples/runs/arxiv", "examples/runs/peptides_func")
RUNTIME_SOURCES = ("cross_morse/core.py", "cross_morse/learning.py", "cross_morse/policies.py",
                   "cross_morse/pipeline.py", "cross_morse/training.py", "cross_morse/exploration.py",
                   "cross_morse/data.py")

S1_GROUP = "supplementary_table_S1_statistical_design"
S2_GROUP = "supplementary_table_S2_data_role_ledger"
S3_GROUP = "supplementary_table_S3_target_regime_contract"
S5_GROUP = "supplementary_table_S5_frozen_model_inventory"
S5B_GROUP = "supplementary_table_S5b_instrumentation_map"
S5D_GROUP = "supplementary_table_S5d_model_inventory"
S8_GROUP = "supplementary_table_S8_online_features"
S9_GROUP = "supplementary_table_S9_observer_and_search"
S10_GROUP = "supplementary_table_S10_policy_definitions"
S11_GROUP = "supplementary_table_S11_confirmation_map"
S24_GROUP = "supplementary_table_S24_claim_artifact_registry"
S31_GROUP = "supplementary_table_S31_reproduction_map"
S32_GROUP = "supplementary_table_S32_comparator_rules"
S35_GROUP = "supplementary_table_S35_reporting_conventions"
S36_GROUP = "supplementary_table_S36_experiment_definitions"
S38_GROUP = "supplementary_table_S38_results_record_schema"
S39_GROUP = "supplementary_table_S39_data_role_contract"
S40_GROUP = "supplementary_table_S40_information_restrictions"

ROLE_ORDER = ("fit", "explore", "select", "confirm", "compare", "admit", "test")
ROLE_LABELS = {role: f"D_{role}" for role in ROLE_ORDER}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


# ---------------------------------------------------------------------------
# Repository sources
# ---------------------------------------------------------------------------

def protocol_text(ctx, stem: str) -> dict:
    """The textual parts of a protocol display, with their page provenance."""
    path = ctx.root / "paper" / "protocol" / f"{stem}.json"
    if not path.is_file():
        raise DisplayInputError(f"{path.relative_to(ctx.root)} is absent")
    record = json.loads(path.read_text(encoding="utf-8"))
    source = record.get("source") or {}
    if not source.get("document") or not source.get("pages") or not isinstance(record.get("rows"), dict):
        raise DisplayInputError(f"paper/protocol/{stem}.json needs source.document, source.pages and rows")
    return record


def fill(text: str, values: dict) -> str:
    """Replace ``{name}`` placeholders; every placeholder must have a value."""
    def value(match):
        if match.group(1) not in values:
            raise DisplayInputError(f"no value for placeholder {{{match.group(1)}}} in {text!r}")
        return str(values[match.group(1)])
    return re.sub(r"\{([a-z_][a-z_0-9]*)\}", value, text)


def configs(ctx) -> dict[str, dict]:
    return {dataset: ctx.config(dataset, PROFILE) for dataset in DATASETS}


def common(values, what: str):
    """The one value that every dataset configuration shares."""
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    if len(distinct) != 1:
        raise DisplayInputError(f"the full configurations disagree on {what}: {sorted(distinct)}")
    return json.loads(distinct.pop())


@functools.lru_cache(maxsize=None)
def _parsed(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)


def source_tree(ctx, relative: str) -> ast.Module:
    return _parsed(str(ctx.root / relative))


def source_text(ctx, relative: str) -> str:
    return (ctx.root / relative).read_text(encoding="utf-8")


def find_class(tree, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise DisplayInputError(f"class {name} is not defined")


def find_function(tree, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise DisplayInputError(f"function {name} is not defined")


def literal(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def keyed_literal(tree, key: str):
    """The literal stored under ``key`` in a dict display of the module (e.g. an audit record)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return ast.literal_eval(v)
    raise DisplayInputError(f"no literal entry {key!r}")


def function_defaults(tree, name: str) -> dict:
    fn = find_function(tree, name)
    args, defaults = fn.args.args, fn.args.defaults
    return {a.arg: ast.literal_eval(d) for a, d in zip(args[len(args) - len(defaults):], defaults)}


def module_layers(tree, class_name: str, attribute: str | None = None) -> list[tuple]:
    """``nn`` layer constructors of a module's ``__init__`` in source order.

    Each layer is ``(name, *literal arguments)``; non-literal arguments are ``None``.
    With ``attribute`` only the layers assigned to ``self.<attribute>`` are returned.
    """
    init = find_function(find_class(tree, class_name), "__init__")
    roots = [init]
    if attribute is not None:
        roots = [node.value for node in ast.walk(init) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == attribute for t in node.targets)]
        if not roots:
            raise DisplayInputError(f"{class_name}.{attribute} is not assigned")
    calls = sorted({id(n): n for root in roots for n in ast.walk(root)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "nn"
                    and n.func.attr != "Sequential"}.values(), key=lambda n: (n.lineno, n.col_offset))
    return [(call.func.attr, *(literal(a) for a in call.args)) for call in calls]


def method_call_literals(tree, class_name: str, method: str, attribute: str) -> tuple:
    """Literal arguments of the first ``.<attribute>(...)`` call inside a method."""
    fn = find_function(find_class(tree, class_name), method)
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attribute:
            return tuple(ast.literal_eval(a) for a in node.args)
    raise DisplayInputError(f"{class_name}.{method} makes no {attribute} call")


@functools.lru_cache(maxsize=None)
def _actions(root: str):
    from cm_theory.operators import load_registry
    return load_registry(Path(root) / "cross_morse" / "core.py")


def action_registry(ctx):
    """The ACTIONS tuple of cross_morse/core.py, read as literals (cm_theory.operators)."""
    return _actions(str(ctx.root))


def route_registry(ctx) -> list[tuple[str, int, bool]]:
    """(route, complexity, is_open_loop) of every ``add`` call of policies.make_registry."""
    fn = find_function(source_tree(ctx, "cross_morse/policies.py"), "make_registry")
    routes: dict[str, tuple[int, bool]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "add" \
                and len(node.args) >= 3:
            route, complexity, open_loop = (ast.literal_eval(a) for a in node.args[:3])
            if routes.get(route, (complexity, open_loop)) != (complexity, open_loop):
                raise DisplayInputError(f"route {route} is registered with two complexities")
            routes[route] = (complexity, open_loop)
    return sorted(((r, c, o) for r, (c, o) in routes.items()), key=lambda item: item[1])


def measurement_schema(ctx, bands, sensing_period: int) -> list[str]:
    """Coordinate names of MeasurementBank (cross_morse/core.py) for one sensor setting.

    The constructor's statements that build ``self.names`` are evaluated literally: the
    cheap list, the per-band and per-action comprehensions and the spectral-age
    coordinate that is appended when the sensing period exceeds one.
    """
    init = find_function(find_class(source_tree(ctx, "cross_morse/core.py"), "MeasurementBank"), "__init__")
    names: list[str] = []

    def is_names(target):
        return isinstance(target, ast.Attribute) and target.attr == "names"

    def evaluate(value):
        if isinstance(value, ast.List):
            return [ast.literal_eval(v) for v in value.elts]
        if isinstance(value, ast.ListComp) and isinstance(value.elt, ast.JoinedStr):
            prefix = "".join(p.value for p in value.elt.values if isinstance(p, ast.Constant))
            source = value.generators[0].iter
            if isinstance(source, ast.Attribute) and source.attr == "bands":
                items = list(bands)
            elif isinstance(source, ast.Call) and getattr(source.func, "id", None) == "range":
                items = list(range(*(ast.literal_eval(a) for a in source.args)))
            else:
                raise DisplayInputError("unrecognized MeasurementBank name comprehension")
            return [f"{prefix}{item}" for item in items]
        raise DisplayInputError("unrecognized MeasurementBank name statement")

    for statement in init.body:
        if isinstance(statement, ast.Assign) and any(is_names(t) for t in statement.targets):
            names = evaluate(statement.value)
        elif isinstance(statement, ast.AugAssign) and is_names(statement.target):
            names += evaluate(statement.value)
        elif isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) \
                and getattr(statement.test.left, "attr", None) == "sensing_period":
            threshold = ast.literal_eval(statement.test.comparators[0])
            for inner in statement.body:
                call = getattr(inner, "value", None)
                if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "append" \
                        and is_names(call.func.value) and sensing_period > threshold:
                    names.append(ast.literal_eval(call.args[0]))
    if not names:
        raise DisplayInputError("MeasurementBank defines no coordinate names")
    return names


def makefile_targets(ctx) -> set[str]:
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", source_text(ctx, "Makefile"), re.M))


def runtime_writes(ctx, relative: str) -> bool:
    """A run-directory path written by the released runtime.

    True when a bundled example run holds it, or when the runtime source joins its
    components literally (``self.out / "admit" / "prediction_execution.json"``), which
    covers outputs newer than the bundled examples.
    """
    if any((ctx.root / run / relative).exists() for run in EXAMPLE_RUNS):
        return True
    joined = " / ".join(f'"{part}"' for part in relative.rstrip("/").split("/"))
    return any(joined in source_text(ctx, path) for path in RUNTIME_SOURCES)


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def sci(value: float) -> str:
    """Learning-rate style: 3e-4, 1e-5, 0.01."""
    if value >= 0.01:
        return format(value, "g")
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def minus(text: str) -> str:
    return text.replace("-", "−")


def braces(values) -> str:
    return "{" + ", ".join(format(v, "g") for v in values) + "}"


def loss_name(cfg: dict) -> str:
    from cross_morse.tasks import task_spec
    return task_spec(cfg).loss_name.upper()


# ---------------------------------------------------------------------------
# Supplementary Table S1: statistical units and admission criteria
# ---------------------------------------------------------------------------

PARTITIONERS = {"metis": "METIS", "degree_stratified": "degree-stratified"}


@display("supplementary:Table S1", stem="S1", title="Dataset statistical units and admission criteria",
         scope="protocol")
def table_s1(ctx, out):
    from cm_campaigns.contract import UNITS
    from cross_morse.evaluation import independent_graphs
    text = protocol_text(ctx, "S1")["rows"]
    sources = {row["dataset"]: row for row in ctx.source_json("data/sources.json")["datasets"]}
    counts = ctx.constants["data_roles_table_S2"]["multi_graph_counts"]
    rows, source = [], []
    for dataset in DATASETS:
        cfg = ctx.config(dataset, PROFILE)
        data, adm = cfg["data"], cfg["admission"]
        independent = independent_graphs(cfg)
        loss = loss_name(cfg)
        margin = f"ε_a = {adm['epsilon_a']:g}"
        metric = f"metric gate (ε_metric = {adm['epsilon_metric']:g})"
        splits = len(sources[dataset].get("split_indices") or [])
        J = counts[dataset]["D_admit"] if independent else None
        if independent:
            primary, repeated = "complete graph", f"J = {J}"
            treatment, role = "graph bootstrap and empirical Bernstein", "inferential"
            criterion = (f"Theorem 2: U_a ≤ −ε_a ({margin}) with B = {adm['clip']:.2f}, δ_a = {adm['alpha']:g}; "
                         f"{metric}; policy fixed on D_confirm before D_admit is read")
        else:
            cores = (f"{data['num_clusters']} {PARTITIONERS[data['partitioner']]} cores; "
                     f"{data['buffer_hops']}-hop buffer")
            rule = (f"clipped-{loss} mean ≤ −ε_a and cluster-bootstrap upper < 0 on D_admit "
                    f"({margin}, B = {adm['clip']:.2f}); {metric}")
            if UNITS[dataset] == "official_split":
                primary, repeated = "official split", f"{splits} splits ({cores} per split)"
                treatment = "paired small-sample split analysis; cluster bootstrap within each split"
                role = "descriptive (repeated splits)"
                criterion = (f"descriptive per split: {rule}; campaign G_admit = 1 only if all {splits} splits "
                             f"admit; paired t interval over the {splits} splits reported")
            else:
                primary = ("degree-stratified buffered cluster" if data["partitioner"] == "degree_stratified"
                           else "buffered graph cluster")
                repeated = cores
                treatment = "cluster bootstrap" + (f" and {text[dataset]['sensitivity']}" if dataset in text else "")
                role, criterion = "descriptive", f"descriptive: {rule}"
        gate = "yes" if independent else "no"
        rows.append([dataset, "many" if independent else "1", primary, repeated, treatment, role, criterion, gate])
        source.append([dataset, independent, UNITS[dataset], data.get("partitioner"), data.get("num_clusters"),
                       data.get("buffer_hops"), splits or None, J, adm["rule"], adm["clip"], adm["alpha"],
                       adm["epsilon_a"], adm["epsilon_metric"], adm["bootstrap_reps"], loss])
        for key, value in (("independent_graphs", "many" if independent else "1"), ("primary_unit", primary),
                           ("repeated_units", repeated), ("interval_role", role), ("formal_gate", gate)):
            out.compare(S1_GROUP, (dataset, key), value)
        if independent:
            for key, value in (("J", J), ("B", adm["clip"]), ("delta_a", adm["alpha"])):
                out.compare(S1_GROUP, (dataset, key), value)
    out.uses("configuration", "configs/*_full.json", ["data", "admission"])
    out.uses("code", "cross_morse", ["evaluation.py:independent_graphs", "statistics.py:admission",
                                     "cm_campaigns/contract.py:UNITS"])
    out.table(["Dataset", "independent graphs", "primary unit", "J or repeated units", "dependence treatment",
               "interval role", "admission criterion for G_admit", "formal gate?"], rows,
              notes=["Rows are read from configs/<dataset>_full.json (data, admission), the statistical units of "
                     "cm_campaigns/contract.py and the admission rule of cross_morse.statistics.admission; J is the "
                     "official validation count of paper/registered_constants.json.",
                     "Roman-empire: every split run admits with the cluster-bootstrap rule on its own cores; the "
                     "campaign record (paper/build/evidence.py) admits only when every split admits."])
    out.data("design", ["dataset", "independent_graphs", "unit", "partitioner", "num_clusters", "buffer_hops",
                        "official_splits", "J_admit", "admission_rule", "clip_B", "delta_a", "epsilon_a",
                        "epsilon_metric", "bootstrap_reps", "loss"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S2: split, fold, seed and decision ledger
# ---------------------------------------------------------------------------

def role_shares() -> dict[str, float]:
    from cross_morse.split_contract import TRAIN_CUTS
    cuts = (0.0, *TRAIN_CUTS, 1.0)
    return dict(zip(ROLE_ORDER[:5], (b - a for a, b in zip(cuts, cuts[1:]))))


def role_sizes(n: int) -> dict[str, int]:
    """Role sizes of ``n`` official training units under the preparation rule of cross_morse/data.py."""
    from cross_morse.split_contract import TRAIN_CUTS
    cuts = [0, *(int(n * f) for f in TRAIN_CUTS), n]
    return dict(zip(ROLE_ORDER[:5], (b - a for a, b in zip(cuts, cuts[1:]))))


def campaign_seed_count(ctx) -> int:
    """Controller-optimization seeds required by cm_experiments.campaign.slots."""
    fn = find_function(source_tree(ctx, "cm_experiments/campaign.py"), "slots")
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call) \
                and getattr(node.left.func, "id", None) == "len" and len(node.comparators) == 1 \
                and isinstance(node.comparators[0], ast.Constant) and isinstance(node.comparators[0].value, int):
            return node.comparators[0].value
    raise DisplayInputError("cm_experiments.campaign.slots declares no seed count")


@display("supplementary:Table S2", stem="S2", title="Data roles, folds, seeds and decisions", scope="protocol")
def table_s2(ctx, out):
    from cm_experiments.campaign import SEED_SCOPE
    text = protocol_text(ctx, "S2")["rows"]
    cfgs = configs(ctx)
    rule = ctx.source_json("data/sources.json")["role_split_rule"]
    shares = role_shares()
    seeds = common([len(c["training"]["seeds"]) for c in cfgs.values()], "training seeds")
    outer = common([c["selection"]["outer_folds"] for c in cfgs.values()], "outer folds")
    inner = common([c["selection"]["inner_folds"] for c in cfgs.values()], "inner folds")
    crossfit = common([c["controller"]["crossfit_folds"] for c in cfgs.values()], "trajectory folds")
    ensembles = common([c["registry"]["ensemble_sizes"] for c in cfgs.values()], "ensemble sizes")
    internal = common([c["data"]["internal_val_fraction"] for c in cfgs.values()], "internal validation fraction")
    controller_seeds = campaign_seed_count(ctx)
    allocation = {role: f"{pct(shares[role])} of official training" for role in ROLE_ORDER[:3]}
    allocation["fit"] += f"; {text['fit']['allocation_note']} ({pct(internal)} of D_fit)"
    for role in ("confirm", "compare"):
        allocation[role] = f"{pct(shares[role])}; {text[role]['allocation_note']}"
    allocation["admit"] = f"{rule['admit']}; {text['admit']['allocation_note']}"
    allocation["test"] = rule["test"]
    folds = {"outer": {"select": outer}, "inner": {"explore": f"{crossfit} trajectory folds", "select": inner}}
    if "frozen_parent_ensemble" not in SEED_SCOPE:
        raise DisplayInputError(f"unexpected E3–E6 seed scope {SEED_SCOPE}")
    plant = {"fit": seeds, "select": f"ensembles E ∈ {braces(ensembles)}", "compare": "frozen parent ensemble"}
    controller = {"explore": f"{seeds} (trajectory t on checkpoint t mod {seeds})",
                  "compare": f"{controller_seeds} (E3–E6 controller optimization)"}
    rows, numbers = [], []
    for role in ROLE_ORDER:
        na = not_applicable(f"The released {ROLE_LABELS[role]} stage does not separately execute this fold or seed loop")
        rows.append([ROLE_LABELS[role], allocation[role], folds["outer"].get(role, na), folds["inner"].get(role, na),
                     plant.get(role, na), controller.get(role, na), text[role]["labels"], text[role]["operation"],
                     text[role]["frozen_after"]])
        values = {"allocation_percent": round(100 * shares[role], 6) if role in shares else None,
                  "outer_folds": outer if role == "select" else None,
                  "inner_folds": {"explore": crossfit, "select": inner}.get(role),
                  "plant_seeds": seeds if role == "fit" else None,
                  "controller_seeds": {"explore": seeds, "compare": controller_seeds}.get(role)}
        numbers.append([role, *values.values()])
        for key, value in values.items():
            out.compare(S2_GROUP, (ROLE_LABELS[role], key), value)
    counts = ctx.constants["data_roles_table_S2"]["multi_graph_counts"]
    sizes = []
    for dataset, row in counts.items():
        n = row["official_train"]
        s = role_sizes(n)
        sizes.append([dataset, n, *s.values()])
        out.compare(S2_GROUP, ("multigraph_role_sizes", dataset), [s["confirm"], s["compare"]])
    out.uses("code", "cross_morse/split_contract.py", ["TRAIN_CUTS"])
    out.uses("configuration", "configs/*_full.json", ["training.seeds", "selection", "controller.crossfit_folds",
                                                      "registry.ensemble_sizes", "data.internal_val_fraction"])
    out.table(["Stage", "frozen allocation", "outer folds", "inner folds", "plant seeds", "controller seeds",
               "labels", "operation", "frozen after stage"], rows,
              notes=["Allocations: TRAIN_CUTS of cross_morse/split_contract.py; admission and test splits: "
                     "data/sources.json; folds and seeds: configs/<dataset>_full.json (identical for all seven).",
                     "Controller seeds: exploration cycles trajectories over the plant checkpoints "
                     "(cross_morse/exploration.py); observer and consequence models are fitted once per profile with "
                     "fixed seeds (cross_morse/learning.py); the E3–E6 campaign declares "
                     f"{controller_seeds} controller-optimization seeds (cm_experiments/campaign.py).",
                     "Multigraph role sizes (n_confirm, n_compare) from the preparation rule int(n · cut): " +
                     "; ".join(f"{row[0]} ({row[5]}, {row[6]})" for row in sizes) + "."])
    out.data("ledger", ["role", "allocation_percent", "outer_folds", "inner_folds", "plant_seeds",
                        "controller_seeds"], numbers)
    out.data("multigraph_role_sizes", ["dataset", "official_train", "fit", "explore", "select", "confirm",
                                       "compare"], sizes)


# ---------------------------------------------------------------------------
# Supplementary Table S3: target-regime contract
# ---------------------------------------------------------------------------

OPERATORS = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}


def gate_operator(ctx, key: str) -> str:
    """The comparison the runtime applies to ``entry[key]`` (training.py, pipeline.py)."""
    pattern = re.compile(r'(<=|>=|<|>)\s*-?\s*(?:entry|thresholds)\["' + key + r'"\]')
    found = {m.group(1) for path in ("cross_morse/training.py", "cross_morse/pipeline.py")
             for m in pattern.finditer(source_text(ctx, path))}
    if len(found) != 1:
        raise DisplayInputError(f"the runtime applies {sorted(found)} to entry.{key}")
    return OPERATORS[found.pop()]


def selected_strata_requirement(ctx) -> float:
    """Read the prospective A0-support requirement from the runtime source."""
    for node in source_tree(ctx, "cross_morse/qualification.py").body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SELECTED_STRATA_FRACTION" for t in node.targets):
            value = ast.literal_eval(node.value)
            if type(value) not in (float, int) or not 0 < value <= 1:
                raise DisplayInputError("invalid runtime selected-strata support requirement")
            return float(value)
    raise DisplayInputError("the runtime declares no selected-strata support requirement")


@display("supplementary:Table S3", stem="S3", title="Target-regime qualification contract", scope="protocol")
def table_s3(ctx, out):
    text = protocol_text(ctx, "S3")["rows"]
    cfgs = configs(ctx)
    entry = common([c["entry"] for c in cfgs.values()], "entry thresholds")
    enforced = common([c["selection"]["enforce_entry_contract"] for c in cfgs.values()], "entry enforcement")
    selected_fraction = selected_strata_requirement(ctx)
    op = {key: gate_operator(ctx, key) for key in entry}
    criteria = {
        "nominal_graph_value": minus(f"U(Δ_plant) {op['plant_margin']} -{entry['plant_margin']:.3f} nats"),
        "depth_qualification": (f"PDD {op['pdd_max']} {100 * entry['pdd_max']:.1f} pp; "
                                f"DIL gap {op['dil_gap_max']} {entry['dil_gap_max']:.3f}"),
        "registry_support": (f"ϕ_registry {op['registry_fraction_min']} {entry['registry_fraction_min']:.2f}; "
                             f"selected strata = {selected_fraction:.2f}"),
        "residual_headroom": minus(f"U(Δ_oracle) {op['headroom_margin']} -{entry['headroom_margin']:.3f} nats"),
    }
    action = ("feedback candidates made infeasible; strongest open-loop q_C retained" if enforced
              else "recorded only (selection.enforce_entry_contract is false)")
    rows = [[text[g]["gate"], text[g]["quantity"], criteria[g], action] for g in criteria]
    numbers = {("nominal_graph_value", "epsilon_plant_nats"): entry["plant_margin"],
               ("depth_qualification", "PDD_pp"): 100 * entry["pdd_max"],
               ("depth_qualification", "DIL_gap_nats"): entry["dil_gap_max"],
               ("registry_support", "phi_registry_min"): entry["registry_fraction_min"],
               ("registry_support", "selected_strata"): selected_fraction,
               ("residual_headroom", "epsilon_headroom_nats"): entry["headroom_margin"]}
    for path, value in numbers.items():
        out.compare(S3_GROUP, path, value,
                    note="prospective A0-support gate under Table S35 convention C28"
                    if path[1] == "selected_strata" else "")
    for gate in criteria:
        out.compare(S3_GROUP, (gate, "failure_action"), action)
    out.uses("configuration", "configs/*_full.json", ["entry", "selection.enforce_entry_contract"])
    out.uses("code", "cross_morse", ["training.py:audit_training", "pipeline.py:entry_completion",
                                     "pipeline.py:_stage_select", "qualification.py:selected_strata_support",
                                     "qualification.py:SELECTED_STRATA_FRACTION"])
    out.table(["Gate", "operational quantity", "prespecified criterion", "failure action"], rows,
              notes=["Thresholds: entry section of configs/<dataset>_full.json (identical for all seven); comparison "
                     "operators as executed by cross_morse/training.py and cross_morse/pipeline.py.",
                     "Selected-strata support follows Table S35 convention C28: A0 must be qualified in every "
                     "registered stratum of the full-profile D_explore model-fit support audit. Empty strata or "
                     "missing A0 cells fail. New runs record the versioned gate in select/entry_complete.json; "
                     "older source-bound entry records do not demonstrate enforcement of this added gate.",
                     "Failure action: with selection.enforce_entry_contract, any failed gate makes every feedback "
                     "candidate infeasible before selection (cross_morse/pipeline.py, _stage_select)."])
    out.data("thresholds", ["gate", "key", "operator", "value"],
             [[g, k, op[k], v] for g, k, v in (("nominal_graph_value", "plant_margin", entry["plant_margin"]),
                                                ("depth_qualification", "pdd_max", entry["pdd_max"]),
                                                ("depth_qualification", "dil_gap_max", entry["dil_gap_max"]),
                                                ("registry_support", "registry_fraction_min",
                                                 entry["registry_fraction_min"]),
                                                ("residual_headroom", "headroom_margin", entry["headroom_margin"]))]
             + [["registry_support", "selected_strata_fraction", "=", selected_fraction]])


# ---------------------------------------------------------------------------
# Model descriptions shared by S5, S5b, S5d
# ---------------------------------------------------------------------------

def architecture(cfg: dict) -> dict:
    from scripts.model_inventory import configured_architecture
    return configured_architecture(cfg)["architecture_widths"]


def designated_sites(ctx, cfg: dict) -> dict:
    """Native and inserted controlled sites under the site arithmetic of DesignatedSitePlant.

    Transport plants act natively at every controlled step. GPS places one native
    block per interval after an encoder boundary site, with a decoder boundary site at
    the end; linear GINE places one native site per equal interval.
    """
    depth, backbone = cfg["model"]["depth"], cfg["model"].get("backbone", "shared_transport")
    if backbone not in ("gps", "linear_gine"):
        return {"native": depth, "inserted": 0, "substeps": 1, "boundary": 0}
    layers = architecture(cfg)["native_layers"]
    boundary = 0
    if backbone == "gps":
        match = re.search(r"\(control_depth - (\d+)\) // native_layers", source_text(ctx, "cross_morse/core.py"))
        if match is None:
            raise DisplayInputError("cross_morse/core.py declares no GPS boundary-site rule")
        boundary = int(match.group(1))
    return {"native": layers, "inserted": depth - layers, "substeps": (depth - boundary) // layers,
            "boundary": boundary}


def backbone_label(cfg: dict) -> str:
    model, widths = cfg["model"], architecture(cfg)
    backbone = model.get("backbone", "shared_transport")
    layers, hidden = widths["encoder_layers"], widths["hidden_dim"]
    if backbone == "gps":
        return f"GraphGPS-style {widths['native_layers']}×{hidden}"
    if backbone == "linear_gine":
        return f"GINE-style {widths['native_layers']}×{hidden}, linear aggregation"
    if backbone == "heterophily":
        return f"heterophily encoder{layers}-{hidden}"
    if backbone == "edge_weighted_transport":
        return f"MLP{layers}-{hidden}; edge-feature aggregation"
    return f"MLP{layers}-{hidden} encoder; transport plant"


def head_label(cfg: dict) -> str:
    from cross_morse.tasks import task_spec
    spec, widths = task_spec(cfg), architecture(cfg)
    kind = ("softmax" if spec.task_type == "multiclass"
            else "sigmoid AP" if spec.metric == "ap" else "sigmoid mean-BCE")
    return (f"two-layer {widths['head_hidden']}-wide {kind} head" if widths["head_hidden"] else f"{kind} head")


def architecture_text(cfg: dict) -> str:
    model, widths = cfg["model"], architecture(cfg)
    backbone = model.get("backbone", "shared_transport")
    if backbone == "gps":
        body = (f"GraphGPS-style {widths['native_layers']}×{widths['hidden_dim']} backbone (feed-forward "
                f"{widths['feedforward_hidden']}, {widths['attention_heads']} heads)")
    elif backbone == "linear_gine":
        body = (f"GINE-style {widths['native_layers']}×{widths['hidden_dim']} backbone with degree-normalized "
                f"linear aggregation ({widths['block_mlp_layers']}-layer update MLP {widths['block_mlp_hidden']})")
    else:
        family = {"shared_transport": "shared transport", "edge_weighted_transport": "edge-weighted transport",
                  "heterophily": "heterophily-aware transport"}[backbone]
        body = (f"MLP{widths['encoder_layers']} encoder ({widths['encoder_hidden']}→{widths['hidden_dim']}); "
                f"{family} step")
    return f"{body}; {head_label(cfg)}"


def optimizer_name(ctx) -> str:
    names = set(re.findall(r"torch\.optim\.([A-Za-z]+)\(", source_text(ctx, "cross_morse/training.py")))
    if len(names) != 1:
        raise DisplayInputError(f"cross_morse/training.py uses optimizers {sorted(names)}")
    return names.pop()


def optimizer_text(ctx, cfg: dict) -> str:
    from cross_morse.evaluation import independent_graphs
    tr = cfg["training"]
    batch = f"batch {tr['graph_batch_size']}; " if independent_graphs(cfg) and "graph_batch_size" in tr else ""
    return (f"{optimizer_name(ctx)} {sci(tr['lr'])}, wd {sci(tr['weight_decay'])}, {tr.get('schedule', 'cosine')}; "
            f"{batch}{tr['epochs']} epochs, patience {tr['patience']}")


def checkpoint_text(cfg: dict, *, per_split: bool) -> str:
    rule = cfg["training"]["checkpoint_rule"]
    n = len(cfg["training"]["seeds"])
    if rule == "min_loss":
        text = f"minimum internal-D_fit validation {loss_name(cfg)}"
    elif rule == "max_ap_bce_tiebreak":
        text = "maximum internal-D_fit validation AP with BCE tie-break"
    else:
        raise DisplayInputError(f"unknown checkpoint rule {rule}")
    return f"{text}; {n} seeds" + (" per split" if per_split else "")


def plant_checkpoints(inventory: dict) -> list[dict]:
    rows = [c for c in inventory["checkpoints"]
            if c.get("prediction_role", "q0" if Path(c["path"]).name.startswith("q0_") else "plant") == "plant"]
    if not rows:
        raise DisplayInputError("the model inventory lists no plant checkpoint")
    return rows


def checkpoint_parameters(row: dict) -> int:
    """Exact parameter count of one inventoried checkpoint.

    Older inventories list only ``named_tensor_elements``; their count scope states
    that the plant has no non-parameter buffers, so the two coincide.
    """
    if "parameter_count" in row:
        return int(row["parameter_count"])
    return int(row["named_tensor_elements"]) - int(row.get("buffer_elements", 0))


def unique(values, what: str):
    distinct = sorted(set(values))
    if len(distinct) != 1:
        raise DisplayInputError(f"{what} differs between checkpoints: {distinct}")
    return distinct[0]


# ---------------------------------------------------------------------------
# Supplementary Table S5: frozen-model inventory and training cost
# ---------------------------------------------------------------------------

@display("supplementary:Table S5", stem="S5", title="Frozen-model inventory and training cost",
         needs=("runs",), scope="measurement")
def table_s5(ctx, out):
    rows, inventory_rows, timing_rows = [], [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        members = view.members()
        cfg = members[0].json("config.json")
        counts, hours, devices = [], [], set()
        for split, member in enumerate(members):
            inventory = member.json("train/model_inventory.json")
            for row in plant_checkpoints(inventory):
                counts.append(checkpoint_parameters(row))
                inventory_rows.append([dataset, split if view.is_campaign else None, row["path"],
                                       checkpoint_parameters(row), row.get("sha256")])
            environment = member.json("environment.json")
            devices.add(environment.get("gpu") or environment.get("device") or "unrecorded device")
            for seed in member.json("config.json")["training"]["seeds"]:
                done = member.json(f"train/plant_seed{seed}_done.json")
                seconds = float(done["seconds_this_attempt"])
                hours.append(seconds / 3600.0)
                timing_rows.append([dataset, split if view.is_campaign else None, seed, seconds,
                                    done.get("epochs_completed")])
        count = unique(counts, f"{dataset} plant parameter count")
        per_seed = statistics.fmean(hours)
        model, training = cfg["model"], cfg["training"]
        rows.append([dataset, architecture_text(cfg), integer(model["depth"]), integer(model["hidden_dim"]),
                     num(count / 1e6, 2), f"{model['precision']} plant", optimizer_text(ctx, cfg),
                     checkpoint_text(cfg, per_split=view.is_campaign),
                     f"{', '.join(sorted(devices))}; {per_seed:.2f} h/seed"])
        compared = {"depth": model["depth"], "hidden": model["hidden_dim"], "parameters_M": count / 1e6,
                    "plant_precision": model["precision"], "learning_rate": training["lr"],
                    "weight_decay": training["weight_decay"], "epochs": training["epochs"],
                    "patience": training["patience"], "seeds": len(training["seeds"]),
                    "hours_per_seed": per_seed}
        for key, value in compared.items():
            out.compare(S5_GROUP, (dataset, key), value)
        out.uses("run", dataset, ["config.json", "environment.json", "train/model_inventory.json",
                                  "train/plant_seed*_done.json"])
    out.table(["Dataset", "architecture / head", "depth", "hidden", "params (M)", "precision",
               "optimizer and schedule", "checkpoint rule", "hardware / time per seed"], rows,
              notes=["Architecture, depth, width, precision, optimizer and checkpoint rule: the sealed config.json of "
                     "each run; parameters: the plant checkpoints of train/model_inventory.json (one count per "
                     "dataset, identical over seeds).",
                     "Time per seed: mean over the plant seeds (and, for Roman-empire, over the ten split runs) of "
                     "seconds_this_attempt in train/plant_seed<s>_done.json, in hours; a resumed seed counts only "
                     "its final attempt. Scheduler GPU-hours are the cm_campaigns/accounting.py ledger.",
                     "Node-level plants train full-graph; training.batch_size is not read by "
                     "cross_morse/training.py (dense updates are chunked at model.dense_chunk_size rows)."])
    out.data("inventory", ["dataset", "split", "checkpoint", "parameters", "sha256"], inventory_rows)
    out.data("training_time", ["dataset", "split", "seed", "seconds_this_attempt", "epochs_completed"], timing_rows)


# ---------------------------------------------------------------------------
# Supplementary Table S5b: native and inserted transport sites
# ---------------------------------------------------------------------------

@display("supplementary:Table S5b", stem="S5b", title="Native and inserted transport sites", scope="protocol")
def table_s5b(ctx, out):
    text = protocol_text(ctx, "S5b")["rows"]
    a0 = action_registry(ctx)[0]
    rows, source = [], []
    for dataset in DATASETS:
        cfg = ctx.config(dataset, PROFILE)
        model = cfg["model"]
        sites = designated_sites(ctx, cfg)
        values = {"depth": model["depth"], "dense_chunk_size": f"{model.get('dense_chunk_size', 0):,}",
                  "substeps": sites["substeps"], "boundary": sites["boundary"],
                  "a0_rho": format(a0.rho, "g"), "a0_nu": f"{a0.nu:.2f}", "a0_alpha": f"{a0.alpha:.2f}"}
        cells = {key: fill(text[dataset][key], values) for key in ("native_call", "depth_index", "a0")}
        rows.append([f"{dataset} ({backbone_label(cfg)})", cells["native_call"], cells["depth_index"], cells["a0"]])
        source.append([dataset, model.get("backbone", "shared_transport"), model["depth"], sites["native"],
                       sites["inserted"], sites["substeps"], sites["boundary"]])
        for key in ("native", "inserted"):
            out.compare(S5B_GROUP, (dataset, f"{key}_sites"), sites[key])
        out.compare(S5B_GROUP, (dataset, "controlled_sites"), model["depth"])
        if model.get("backbone") in ("gps", "linear_gine"):
            out.compare(S5B_GROUP, (dataset, "substeps"), sites["substeps"])
    out.compare(S5B_GROUP, ("ogbn-arxiv", "native_a0"), [a0.rho, a0.nu, a0.alpha])
    out.uses("configuration", "configs/*_full.json", ["model"])
    out.uses("code", "cross_morse/core.py", ["ACTIONS", "DesignatedSitePlant"])
    out.table(["Dataset (backbone)", "native call wrapped by the adapter",
               "controlled depth index ℓ versus native layers", "A0 reproduces native aggregation"], rows,
              notes=["Depths, widths and backbones: configs/<dataset>_full.json; native/inserted site counts: the "
                     "site rule of DesignatedSitePlant in cross_morse/core.py; native A0 tuple: the ACTIONS "
                     "registry. Whether A0 reproduces the native aggregation is tested by E1 (Table S5c)."])
    out.data("sites", ["dataset", "backbone", "controlled_sites", "native_sites", "inserted_sites",
                       "substeps_per_native_site", "boundary_sites"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S5d: executable model inventory and reconciliation
# ---------------------------------------------------------------------------

FAMILIES = {"shared_transport": "shared", "edge_weighted_transport": "edge-weighted",
            "heterophily": "heterophily-aware", "gps": "GraphGPS-style local branch", "linear_gine": "linear-GINE"}


def oxford(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


@display("supplementary:Table S5d", stem="S5d", title="Executable model inventory and reconciliation",
         needs=("runs",), scope="measurement")
def table_s5d(ctx, out):
    status = ctx.constants["backbones_source_status"]
    default = status["default_width_plant"]["ogbn-arxiv"]
    commands = next(d["commands"] for d in ctx.manifest["displays"] if d["id"] == "supplementary:Table S5d")
    compare_command = next(c for c in commands if "compare" in c)
    plants, sensors, identity = [], [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        member_counts, natives, inserted = [], set(), set()
        for member in view.members():
            sealed = member.json("config.json")
            inventory = member.json("train/model_inventory.json")
            member_counts += [checkpoint_parameters(r) for r in plant_checkpoints(inventory)]
            for row in member.json("train/manifest.json")["checkpoints"]:
                if not row.get("baseline"):
                    natives.add(row["instrumentation"]["native_site_count"])
                    inserted.add(row["instrumentation"]["inserted_site_count"])
            for record in member.json("train/entry_initial.json")["nominal_identity"]:
                identity.append([dataset, record["seed"], record["max_relative_state_error"],
                                 record["state_tolerance"], record["absolute_fit_val_loss_deviation"],
                                 record["loss_tolerance"], bool(record["pass"]), sealed["model"]["precision"]])
            sensors.append([dataset, sealed["sensor"]["sketch_dim"], sealed["sensor"]["sensing_period"],
                            sealed["controller"]["observer_period"]])
        count = unique(member_counts, f"{dataset} plant parameter count")
        plants.append([dataset, count, unique(natives, "native sites"), unique(inserted, "inserted sites"),
                       len(member_counts)])
        out.uses("run", dataset, ["config.json", "train/model_inventory.json", "train/manifest.json",
                                  "train/entry_initial.json"])
    arxiv_view = ctx.require_run("ogbn-arxiv")
    arxiv_cfg = arxiv_view.members()[0].json("config.json")
    arxiv_inventory = arxiv_view.members()[0].json("train/model_inventory.json")
    widths = architecture(arxiv_cfg)
    full = plants[0][1]
    benchmark = (f"MLP{widths['encoder_layers']}-{widths['hidden_dim']} encoder; "
                 f"{arxiv_cfg['model']['depth']} unrolled transport steps")
    dims = (arxiv_inventory.get("input_dim") or status["released_full_profiles"]["ogbn-arxiv"]["input_dim"],
            arxiv_inventory.get("num_outputs") or status["released_full_profiles"]["ogbn-arxiv"]["num_outputs"])
    present = {ctx.require_run(d).members()[0].json("config.json")["model"].get("backbone", "shared_transport")
               for d in DATASETS[1:]}
    families = oxford([label for backbone, label in FAMILIES.items() if backbone in present])
    others = families[0].upper() + families[1:] + " families"
    observer = sorted({row[3] for row in sensors})
    sensor_fields = sorted(k for k in ("sensing_period", "sketch_dim") if k in arxiv_cfg["sensor"])
    executed = "; ".join(f"{SHORT[d]} ({k}, {p})" for d, k, p, _ in
                         {row[0]: row for row in sensors}.values())
    passed = sum(row[6] for row in identity)
    rows = [
        ["ogbn-arxiv default plant", f"{default:,} parameters in the released default configuration",
         f"The benchmark plant of Table S5 is the full-profile configuration ({benchmark.replace(';', ',')}, "
         f"{full / 1e6:.2f} M parameters); the instantiated architecture and exact parameter count of every "
         f"checkpoint are read from the train/model_inventory.json of the run archive ({plants[0][4]} plant "
         f"checkpoints, {full:,} parameters each), which {compare_command} reconciles with Table S5."],
        ["Other dataset plants", others,
         "Exact parameter counts and native/inserted-site maps of every instantiated configuration: " +
         "; ".join(f"{SHORT[d]} {c:,} ({n} native / {i} inserted sites)" for d, c, n, i, _ in plants[1:]) +
         f"; reconciled with Table S5 by {compare_command}."],
        ["Sensor", f"Explicit {' and '.join(sensor_fields)}; observer update period "
                   f"(controller.observer_period = {', '.join(map(str, observer))}) logged separately",
         f"Sensor settings sealed in each run's config.json; executed (sketch_dim, sensing_period): {executed}."],
        ["Execution identity", "Native step versus A0 on identical internal D_fit validation inputs",
         f"Maximum relative state deviation and loss deviation against the dtype tolerances, per checkpoint "
         f"(train/entry_initial.json): {passed} of {len(identity)} checkpoints pass; largest state deviation "
         f"{max(r[2] for r in identity):.1e}, largest loss deviation {max(r[4] for r in identity):.1e}."],
    ]
    compared = {
        ("ogbn-arxiv", "released_default_parameters"): default,
        ("ogbn-arxiv", "benchmark_parameters_M"): full / 1e6,
        ("ogbn-arxiv", "benchmark_architecture"): benchmark,
        ("ogbn-arxiv", "inventory_artifact"): "train/model_inventory.json",
        ("ogbn-arxiv", "comparison_command"): compare_command,
        ("ogbn-arxiv", "evidence_status"): "instantiated_source_count",
        ("ogbn-arxiv", "full_profile_parameters"): full,
        ("ogbn-arxiv", "released_default_scope"): (
            f"plant built with the default widths (encoder width = hidden width, linear head) at {dims[0]} inputs "
            f"and {dims[1]} outputs"),
        ("ogbn-arxiv", "full_profile_scope"): (
            f"configs/{STEMS['ogbn-arxiv']}_full.json (encoder width {widths['encoder_hidden']:,})"),
        ("other_plants",): others,
        ("sensor_fields",): sensor_fields,
        ("observer_period",): rows[2][1],
        ("execution_identity",): rows[3][1],
    }
    for path, value in compared.items():
        prose = path[-1] in ("released_default_scope", "full_profile_scope", "other_plants", "observer_period",
                             "execution_identity")
        out.compare(S5D_GROUP, path, value, note="generated prose; wording is not transcribed" if prose else "")
    out.table(["Item", "Executable specification", "Evidence and status"], rows,
              notes=["Default-width count: the instantiation recorded in paper/registered_constants.json "
                     "(backbones_source_status.default_width_plant; scripts/model_inventory.py --default-widths). "
                     "All other counts, site maps, sensor settings and identity checks are read from the runs."])
    out.data("plants", ["dataset", "parameters", "native_sites", "inserted_sites", "plant_checkpoints"], plants)
    out.data("sensor", ["dataset", "sketch_dim", "sensing_period", "observer_period"], sensors)
    out.data("identity", ["dataset", "seed", "max_relative_state_error", "state_tolerance",
                          "absolute_fit_val_loss_deviation", "loss_tolerance", "pass", "precision"], identity)


# ---------------------------------------------------------------------------
# Supplementary Table S8: online features and observer diagnostics
# ---------------------------------------------------------------------------

def row_coordinates(spec: dict, schema: list[str]) -> list[str]:
    if "coordinate_prefix" in spec:
        return [name for name in schema if name.startswith(spec["coordinate_prefix"])]
    return [name for name in spec.get("coordinates", []) if name in schema]


@display("supplementary:Table S8", stem="S8", title="Online features and observer diagnostics", scope="protocol")
def table_s8(ctx, out):
    text = protocol_text(ctx, "S8")["rows"]
    cfgs = configs(ctx)
    core = source_tree(ctx, "cross_morse/core.py")
    observer_dim = keyed_literal(core, "observer_innovation_features_external")
    normalization = keyed_literal(source_tree(ctx, "cross_morse/learning.py"), "implementation_choices")["normalization"]
    clip = re.search(r"clip\[(-?[\d.]+),\s*(-?[\d.]+)\]", normalization)
    if clip is None:
        raise DisplayInputError("cross_morse/learning.py records no normalization clip")
    bands = common([c["sensor"]["bands"] for c in cfgs.values()], "sensor bands")
    observer_period = common([c["controller"]["observer_period"] for c in cfgs.values()], "observer period")
    schemas = {d: measurement_schema(ctx, c["sensor"]["bands"], c["sensor"]["sensing_period"])
               for d, c in cfgs.items()}
    widest = max(schemas.values(), key=len)
    periods = {"cheap": "every depth", "spectral": "sensing period P_s", "reach": "every depth; powers at P_s",
               "age": "every depth", "observer": f"every update (observer period r = {observer_period})"}
    rows, mapping = [], []
    kind_of: dict[str, str] = {}
    for key, spec in text.items():
        if spec["kind"] == "observer":
            dim, names = observer_dim, ["observer innovation (external to the sensor)"]
        else:
            names = row_coordinates(spec, widest)
            dim = len(names)
            kind_of.update(dict.fromkeys(names, spec["kind"]))
        transform = spec["transform"]
        if key == "entropy":
            transform = f"{transform} [{minus(clip.group(1))}, {minus(clip.group(2))}]"
        rows.append([spec["measurement"], spec["formula"], integer(dim), transform, periods[spec["kind"]],
                     spec["failure"]])
        mapping.append([key, spec["kind"], dim, " ".join(names)])
        out.compare(S8_GROUP, (key, "dim"), dim)
        out.compare(S8_GROUP, (key, "period"), periods[spec["kind"]])
    unclaimed = sorted(set(widest) - set(kind_of))
    if unclaimed:
        raise DisplayInputError(f"S8 rows do not cover the sensor coordinates {unclaimed}")
    cheap = sum(kind == "cheap" for kind in kind_of.values())
    spectral = sum(kind in ("spectral", "reach") for kind in kind_of.values())
    totals = sorted({len(s) for s in schemas.values()})
    out.compare(S8_GROUP, ("coordinates", "cheap"), cheap)
    out.compare(S8_GROUP, ("coordinates", "spectral_and_reach"), spectral)
    out.compare(S8_GROUP, ("coordinates", "totals"), totals)
    out.uses("code", "cross_morse/core.py", ["MeasurementBank.__init__", "MeasurementBank.metadata"])
    out.uses("code", "cross_morse/learning.py", ["implementation_choices.normalization"])
    out.uses("configuration", "configs/*_full.json", ["sensor", "controller.observer_period"])
    out.table(["measurement", "exact formula/source", "dim.", "normalization", "period", "failure targeted"], rows,
              notes=[f"Coordinates: MeasurementBank of cross_morse/core.py with bands {braces(bands)}: {cheap} cheap "
                     f"coordinates at every site and {spectral} spectral and reach coordinates; a sensing period "
                     f"above one adds the spectral age, for {' or '.join(map(str, totals))} coordinates.",
                     "Sensing periods P_s of the full configurations: " +
                     ", ".join(f"{SHORT[d]} {c['sensor']['sensing_period']}" for d, c in cfgs.items()) + ".",
                     f"Robust normalization (cross_morse/learning.py): {normalization}. Innovation and NIS are "
                     f"computed by the observer ({observer_dim} coordinates) and are not sensor coordinates."])
    out.data("coordinates", ["row", "kind", "dim", "coordinates"], mapping)
    out.data("schemas", ["dataset", "sensing_period", "dimension", "coordinates"],
             [[d, cfgs[d]["sensor"]["sensing_period"], len(s), " ".join(s)] for d, s in schemas.items()])


# ---------------------------------------------------------------------------
# Supplementary Table S9: observer, consequence-model and search configuration
# ---------------------------------------------------------------------------

def critic_text(ctx) -> tuple[str, dict]:
    learning = source_tree(ctx, "cross_morse/learning.py")
    layers = module_layers(learning, "ActionCritic")
    linear = [layer for layer in layers if layer[0] == "Linear"]
    activations = sorted({layer[0] for layer in layers if layer[0] not in ("Linear", "Dropout")})
    dropouts = [layer[1] for layer in layers if layer[0] == "Dropout"]
    low, high = method_call_literals(learning, "ActionCritic", "forward", "clamp")
    hidden = [layer[2] for layer in linear[:-1]]
    outputs = linear[-1][2]
    text = (f"MLP {'–'.join(map(str, hidden))}, {'/'.join(activations)}; dropout {unique(dropouts, 'dropout'):.2f} "
            f"after the first {NUMBER_WORDS[len(dropouts)]} hidden layers; "
            f"{'mean and log-scale outputs' if outputs == 2 else f'{outputs} outputs'}, log-scale clipped to "
            f"[{minus(format(low, 'g'))}, {format(high, 'g')}]")
    return text, {"hidden_widths": hidden, "dropout": dropouts[0], "log_scale_clip": [low, high]}


def sequence_model_text(ctx) -> tuple[str, dict]:
    learning = source_tree(ctx, "cross_morse/learning.py")
    entry = module_layers(learning, "ResidualModel", "input")
    residual = module_layers(learning, "ResidualModel", "residual")
    output = module_layers(learning, "ResidualModel", "output")
    width = entry[0][2]
    blocks = [layer for layer in residual if layer[0] == "Linear"]
    activation = sorted({layer[0] for layer in residual if layer[0] not in ("Linear", "Dropout")})
    text = (f"{width}-wide input, {NUMBER_WORDS[len(blocks)]} {unique([b[2] for b in blocks], 'residual width')}-wide "
            f"residual layers, {'/'.join(activation)} and a {'linear' if output[0][0] == 'Linear' else output[0][0]} "
            f"output layer")
    return text, {"width": width, "residual_layers": len(blocks)}


@display("supplementary:Table S9", stem="S9", title="Observer, consequence-model and search configuration",
         scope="protocol")
def table_s9(ctx, out):
    text = protocol_text(ctx, "S9")["rows"]
    cfgs = configs(ctx)
    controller = common([c["controller"] for c in cfgs.values()], "controller settings")
    sensor = common([{k: c["sensor"][k] for k in ("bands", "degree")} for c in cfgs.values()], "sensor bands")
    ensembles = common([c["registry"]["ensemble_sizes"] for c in cfgs.values()], "ensemble sizes")
    clip = common([c["admission"]["clip"] for c in cfgs.values()], "loss clipping")
    grid = [b for b in ctx.constants["sensitivity_grids_table_S18"]["clip_B"] if not math.isclose(b, clip)]
    windows = function_defaults(source_tree(ctx, "cross_morse/core.py"), "chebyshev_window_coefficients")["windows"]
    observer_doc = ast.get_docstring(find_class(source_tree(ctx, "cross_morse/learning.py"), "ConditionedObserver"))
    observer_kind = observer_doc.splitlines()[0].rstrip(".")
    critic, critic_values = critic_text(ctx)
    sequence, sequence_values = sequence_model_text(ctx)
    values = {
        "trajectory_folds": str(controller["crossfit_folds"]),
        "observer": f"{observer_kind}; {controller['state_dim']} dimensions, rank {controller['covariance_rank']}",
        "filters": f"{windows} candidates, {len(sensor['bands'])} active, degree {sensor['degree']}",
        "critic": critic, "sequence_models": sequence,
        "search": f"{braces(controller['horizons'])}/{controller['beam_width']}/{controller['top_k']}",
        "observer_period": f"r={controller['observer_period']}",
        "ensemble": f"E ∈ {braces(ensembles)}",
        "clipping": f"primary B = {clip:.2f}; sensitivity B ∈ {{{', '.join(f'{b:.2f}' for b in grid)}}}",
    }
    rows = [[text[key]["object"], values[key], text[key]["rule"]] for key in text]
    compared = {("trajectory_folds", "value"): controller["crossfit_folds"],
                ("observer", "state_dim"): controller["state_dim"],
                ("observer", "rank"): controller["covariance_rank"],
                ("filters", "candidates"): windows, ("filters", "active"): len(sensor["bands"]),
                ("filters", "degree"): sensor["degree"],
                **{("critic", k): v for k, v in critic_values.items()},
                **{("sequence_models", k): v for k, v in sequence_values.items()},
                ("search", "horizons"): controller["horizons"], ("search", "beam"): controller["beam_width"],
                ("search", "top_k"): controller["top_k"], ("observer_period", "r"): controller["observer_period"],
                ("ensemble", "sizes"): ensembles, ("clipping", "B"): clip, ("clipping", "sensitivity"): grid}
    for path, value in compared.items():
        out.compare(S9_GROUP, path, value)
    out.uses("configuration", "configs/*_full.json", ["controller", "sensor", "registry.ensemble_sizes",
                                                      "admission.clip"])
    out.uses("code", "cross_morse", ["learning.py:ConditionedObserver", "learning.py:ActionCritic",
                                     "learning.py:ResidualModel", "core.py:chebyshev_window_coefficients"])
    out.table(["object", "implemented primary value", "required sensitivity or failure rule"], rows,
              notes=["Primary values: configs/<dataset>_full.json (identical for all seven) and the observer, critic and "
                     "residual-model definitions of cross_morse/learning.py; candidate windows: "
                     "chebyshev_window_coefficients in cross_morse/core.py; clipping sensitivity: "
                     "sensitivity_grids_table_S18 of paper/registered_constants.json."])
    out.data("values", ["object", "key", "value"],
             [[path[0], path[1], json.dumps(value)] for path, value in compared.items()])


# ---------------------------------------------------------------------------
# Supplementary Table S10: executable policy definitions
# ---------------------------------------------------------------------------

@display("supplementary:Table S10", stem="S10", title="Executable policy definitions", scope="protocol")
def table_s10(ctx, out):
    text = protocol_text(ctx, "S10")["rows"]
    routes = route_registry(ctx)
    if [r for r, _, _ in routes] != list(text):
        raise DisplayInputError(f"S10 rows {list(text)} differ from the registered routes {[r for r, _, _ in routes]}")
    rows = []
    for route, complexity, open_loop in routes:
        spec = text[route]
        feedback = "no" if open_loop else "yes"
        rows.append([spec["route"], spec["input"], spec["decision_time"], spec["objective"], spec["constraints"],
                     spec["output"], feedback])
        out.compare(S10_GROUP, (route, "feedback"), feedback)
    out.uses("code", "cross_morse/policies.py", ["make_registry"])
    out.table(["route", "input", "decision time", "objective", "constraints/failure", "output", "feedback?"], rows,
              notes=["Route set, order (registered complexity) and feedback flag (is_open_loop): make_registry in "
                     "cross_morse/policies.py."])
    out.data("routes", ["route", "complexity", "is_open_loop"], [list(r) for r in routes])


# ---------------------------------------------------------------------------
# Supplementary Table S11: frozen confirmation map
# ---------------------------------------------------------------------------

def confirmation_predecessors(ctx) -> dict[str, str]:
    """Predecessor of each confirmation kind in _stage_confirm (cross_morse/pipeline.py)."""
    fn = find_function(source_tree(ctx, "cross_morse/pipeline.py"), "_stage_confirm")
    found, state = None, None
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            value = literal(node.iter)
            if value and all(isinstance(t, tuple) and len(t) == 3 for t in value):
                found = {kind: route or profile for kind, route, profile in value}
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
            keywords = {kw.arg: kw.value for kw in node.keywords}
            if literal(keywords.get("kind", ast.Constant(None))) == "state":
                state = getattr(keywords.get("predecessor_index"), "id", None)
    if found is None or state != "qC":
        raise DisplayInputError("cross_morse/pipeline.py: confirmation predecessors not found")
    return {"state": "comparator", **found}


def confirmation_margins(kinds) -> dict[str, dict]:
    """Practical margin and beneficial direction of each kind, as cross_morse.statistics.confirmation applies them."""
    from cross_morse.statistics import confirmation
    result = confirmation(0, 0, [], [dict(kind=k, candidate_index=0, predecessor_index=0, available=False)
                                     for k in kinds])
    return {row["kind"]: {"margin": row["practical_margin"], "direction": row["beneficial_direction"]}
            for row in result["contrasts"]}


@display("supplementary:Table S11", stem="S11", title="Frozen confirmation map", scope="protocol")
def table_s11(ctx, out):
    record = protocol_text(ctx, "S11")
    text, labels = record["rows"], record["labels"]
    predecessors = confirmation_predecessors(ctx)
    margins = confirmation_margins(list(text))
    rows, source = [], []
    for kind, spec in text.items():
        margin = margins[kind]["margin"]
        predecessor = predecessors[kind]
        if predecessor == "comparator":
            fail = labels["comparator"]
        else:
            fail = f"{labels[predecessor]} if {labels[predecessor]} ∈ Q_s, otherwise q_C"
        rows.append([spec["route"], spec["contrast"], f"{margin:g} {spec['unit']}", spec["pass"], fail])
        source.append([kind, margin, margins[kind]["direction"], spec["unit"], predecessor])
        out.compare(S11_GROUP, (kind, "margin"), margin)
        out.compare(S11_GROUP, (kind, "fail_output"), fail)
    out.uses("code", "cross_morse", ["statistics.py:confirmation", "pipeline.py:_stage_confirm"])
    out.table(["provisional route", "held-out contrast", "practical margin", "pass output",
               "fail/equality output"], rows,
              notes=["Margins and beneficial directions: cross_morse.statistics.confirmation; predecessors: "
                     "_stage_confirm of cross_morse/pipeline.py (the state contrast is taken against q_C); a failed "
                     "contrast downgrades to its predecessor when that configuration belongs to Q_s, otherwise to "
                     "q_C. A retained route must be Holm-significant against zero and exceed the margin."])
    out.data("map", ["kind", "practical_margin", "beneficial_direction", "unit", "predecessor"], source)


# ---------------------------------------------------------------------------
# Repository linkage shared by S24, S31, S35
# ---------------------------------------------------------------------------

def record_sets(ctx) -> dict[str, dict]:
    """Record-set contracts registered in records/specs.json."""
    return dict(ctx.record_specs["sets"])


def covering_contracts(ctx, displays, experiments=()) -> list[str]:
    """Record sets whose manuscript_tables include one of the displays.

    When both the row and a set declare experiments, they must share one, so that a
    multi-panel figure links only the contract of the panel's experiment.
    """
    labels = {display_id.split(":", 1)[1] for display_id in displays}
    found = []
    for name, spec in record_sets(ctx).items():
        tables = {re.sub(r"^(Supplementary|Main|main) ", "", t) for t in spec.get("manuscript_tables", [])}
        declared = set(spec.get("experiments") or [])
        if labels & tables and (not experiments or not declared or declared & set(experiments)):
            found.append(name)
    return found


def resolved_links(ctx, spec: dict) -> dict:
    """A linkage row, plus the record contracts that cover the displays of a record the code does not produce."""
    if not spec.get("absent"):
        return spec
    extra = [name for name in covering_contracts(ctx, spec.get("displays", []), spec.get("experiments", []))
             if name not in spec.get("records", [])]
    return {**spec, "records": [*spec.get("records", []), *extra]}


def check_linkage(ctx, spec: dict, where: str) -> list[str]:
    """Verify every artifact a linkage row names; return the problems found."""
    problems = []
    sets = record_sets(ctx)
    displays = {d["id"] for d in ctx.manifest["displays"]}
    for path in spec.get("run", []):
        if not runtime_writes(ctx, path):
            problems.append(f"{where}: run artifact {path} is not written by the runtime")
    for name in spec.get("records", []):
        if name not in sets:
            problems.append(f"{where}: record set {name} is not defined in records/specs.json")
    for path in [*spec.get("module", []), *spec.get("campaign", []), *spec.get("repository", [])]:
        if not (ctx.root / path).exists():
            problems.append(f"{where}: {path} does not exist")
    for display_id in spec.get("displays", []):
        if display_id not in displays:
            problems.append(f"{where}: {display_id} is not in paper/display_manifest.json")
    return problems


def linkage_text(spec: dict) -> str:
    parts = []
    if spec.get("run"):
        parts.append("run archive: " + ", ".join(spec["run"]))
    if spec.get("records"):
        parts.append(", ".join(f"records/{name}" for name in spec["records"]))
    if spec.get("campaign"):
        parts.append("campaign export: " + ", ".join(spec["campaign"]))
    if spec.get("module"):
        parts.append(", ".join(spec["module"]))
    if spec.get("absent"):
        parts.append("not produced by the released code: " + "; ".join(spec["absent"]))
    return "; ".join(parts) if parts else MISSING_TEXT


def linkage_rows(ctx, rows: dict) -> list[list]:
    """One row per linked artifact: (row, kind, item, verified); absent records are not verifiable."""
    out = []
    for key, spec in rows.items():
        spec = resolved_links(ctx, spec)
        for kind in ("run", "records", "campaign", "module", "repository", "absent"):
            for item in spec.get(kind, []):
                verified = None if kind == "absent" else not check_linkage(ctx, {kind: [item]}, key)
                out.append([key, kind, item, verified])
    return out


def raise_problems(problems: list[str]) -> None:
    if problems:
        raise DisplayInputError("; ".join(problems))


def placeholder_values(ctx) -> dict:
    """Numbers quoted inside S31/S35 prose, read from configuration, constants and code."""
    from cm_campaigns.contract import HPO_TRIALS
    cfgs = configs(ctx)
    tolerances = training_tolerances(ctx)
    eps_eq = common([c["selection"]["epsilon_eq"] for c in cfgs.values()], "epsilon_eq")
    alternative = sorted(v for v in ctx.constants["sensitivity_grids_table_S18"]["equivalence_margin_epsilon_eq"]
                         if v < eps_eq)
    kappa = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]["ensemble_law_kappa_band"]
    orders = {a.order for a in action_registry(ctx) if a.eta is not None}
    holm = confirmation_alpha()
    return {"epsilon_eq": format(eps_eq, "g"), "epsilon_eq_alternative": format(alternative[-1], "g"),
            "kappa_low": format(kappa[0], "g"), "kappa_high": format(kappa[1], "g"),
            "hpo_trials": HPO_TRIALS, "resolvent_order": unique(orders, "resolvent order"),
            "holm_alpha": format(holm, "g"), **tolerances}


def confirmation_alpha() -> float:
    import inspect
    from cross_morse.statistics import confirmation
    return inspect.signature(confirmation).parameters["alpha"].default


def training_tolerances(ctx) -> dict:
    """E1 tolerances of cross_morse/training.py (state by dtype, loss)."""
    source = source_text(ctx, "cross_morse/training.py")
    state = re.search(r'state_tolerance = ([\d.e-]+) if config\["model"\]\["precision"\] == "bf16" else ([\d.e-]+)',
                      source)
    loss = re.search(r'loss_tolerance = ([\d.e-]+) if spec\.dataset == "Roman-empire" else ([\d.e-]+)', source)
    if state is None or loss is None:
        raise DisplayInputError("cross_morse/training.py declares no E1 tolerances")

    def power(text):
        value = float(text)
        exponent = math.floor(math.log10(value))
        mantissa = value / 10 ** exponent
        return f"10^{exponent}" if math.isclose(mantissa, 1) else f"{mantissa:g} × 10^{exponent}"
    return {"bf16_state_tolerance": power(state.group(1)), "fp32_state_tolerance": power(state.group(2)),
            "loss_tolerance": power(loss.group(2))}


# ---------------------------------------------------------------------------
# Supplementary Table S24: claim-to-artifact registry
# ---------------------------------------------------------------------------

@display("supplementary:Table S24", stem="S24", title="Claim-to-artifact registry", scope="protocol")
def table_s24(ctx, out):
    text = protocol_text(ctx, "S24")["rows"]
    targets = makefile_targets(ctx)
    problems, rows, links = [], [], []
    for key, spec in text.items():
        problems += check_linkage(ctx, spec, key)
        problems += [f"{key}: make {c} is not a Makefile target" for c in spec["checks"] if c not in targets]
        carriers = [*spec.get("run", []), *spec.get("repository", [])]
        rows.append([spec["object"], f"{spec['identity']} ({', '.join(carriers)})", spec["ambiguity"],
                     f"{spec['test']} ({', '.join('make ' + c for c in spec['checks'])})"])
        links += [[key, "artifact", a] for a in carriers]
        links += [[key, "check", f"make {c}"] for c in spec["checks"]]
        for field in ("identity", "ambiguity", "test"):
            out.compare(S24_GROUP, [key, field], spec[field])
    raise_problems(problems)
    out.uses("repository", "linkage", ["paper/protocol/S24.json", "Makefile", *EXAMPLE_RUNS])
    out.table(["object", "required identity", "prohibited ambiguity", "reconstruction test"], rows,
              notes=["Each identity names the run-archive files (relative to a run directory) or repository paths "
                     "that carry it; each reconstruction test names the make target that performs it. Every path is "
                     "written by the released runtime and every target exists in the Makefile.",
                     "Synthetic examples exercise the execution paths and are kept out of the benchmark tables."])
    out.data("linkage", ["object", "kind", "item"], links)


# ---------------------------------------------------------------------------
# Supplementary Table S31: reproduction map
# ---------------------------------------------------------------------------

@display("supplementary:Table S31", stem="S31", title="Reproduction map", scope="protocol")
def table_s31(ctx, out):
    text = protocol_text(ctx, "S31")["rows"]
    values = placeholder_values(ctx)
    problems, rows = [], []
    for key, spec in text.items():
        links = resolved_links(ctx, spec)
        problems += check_linkage(ctx, links, key)
        record = linkage_text(links)
        description = spec.get("record_description")
        if not isinstance(description, str) or not description.strip():
            raise DisplayInputError(f"S31/{key}: the source record description is missing")
        rows.append([fill(spec["quantity"], values), fill(spec["estimator"], values), description, record])
        out.compare(S31_GROUP, (key, "record"), description)
    raise_problems(problems)
    absent = [key for key, spec in text.items() if spec.get("absent")]
    out.uses("repository", "linkage", ["paper/protocol/S31.json", "records/specs.json", "paper/display_manifest.json",
                                       *EXAMPLE_RUNS])
    out.table(["quantity (location)", "estimator", "record", "repository location"], rows,
              notes=["The record column preserves the protocol description in the source PDF. The additional "
                     "repository-location column resolves that description to checked implementation files; "
                     "relocated paths are not treated as changed numerical results.",
                     "Records are the artifacts of this repository: run-archive files written by the runtime, "
                     "measurement-record sets (records/specs.json) and the experiment code that produces them "
                     "(release/experiment_code.json), campaign exports and analytical modules; every path was "
                     "checked to exist.",
                     *([f"Rows whose record the released code does not produce: {', '.join(absent)}; a record "
                        "contract that covers the row's display is linked where one is declared."] if absent else [])])
    out.data("linkage", ["row", "kind", "item", "verified"], linkage_rows(ctx, text))


# ---------------------------------------------------------------------------
# Supplementary Table S32: budget-matched comparator definitions
# ---------------------------------------------------------------------------

def arm_checks(ctx, rows: dict) -> list[str]:
    from cm_experiments.protocol import ARMS
    problems = [f"{key}: arm {spec['arm']} is not in cm_experiments.protocol.ARMS"
                for key, spec in rows.items() if spec.get("arm") and spec["arm"] not in ARMS]
    problems += [f"{key}: {spec['implementation']} does not exist" for key, spec in rows.items()
                 if not (ctx.root / spec["implementation"]).is_file()]
    missing = set(ARMS) - {spec.get("arm") for spec in rows.values()}
    problems += [f"arm {arm} has no row" for arm in sorted(missing)]
    return problems


@display("supplementary:Table S32", stem="S32", title="Budget-matched comparator definitions", scope="protocol")
def table_s32(ctx, out):
    from cm_experiments.budget import Certificate
    record = protocol_text(ctx, "S32")
    text = record["rows"]
    raise_problems(arm_checks(ctx, text))
    tau = Certificate.__dataclass_fields__["floor"].default
    ensembles = common([c["registry"]["ensemble_sizes"] for c in configs(ctx).values()], "ensemble sizes")
    values = {"tau": format(tau, "g"), "ensemble_grid": braces(ensembles)}
    rows = [[spec["policy"], spec["information"], fill(spec["completion"], values)] for spec in text.values()]
    out.compare(S32_GROUP, ("budget", "tau"), tau)
    out.uses("code", "cm_experiments", ["protocol.py:ARMS", "budget.py:Certificate"])
    out.table(["Policy", "information available at decision time", "completion and budget rule"], rows,
              notes=["τ: the certificate floor of cm_experiments.budget.Certificate; ensemble grid: registry.ensemble_sizes "
                     "of the full configurations; arm identifiers: cm_experiments.protocol.ARMS.",
                     "The printed MPC-OL label also names the datasets whose selected comparator is MPC-OL; that is a "
                     "run outcome (select/decision.json), not part of the protocol."])
    out.data("arms", ["row", "arm", "implementation"],
             [[k, s.get("arm"), s["implementation"]] for k, s in text.items()])


# ---------------------------------------------------------------------------
# Supplementary Table S35: reporting conventions and record linkage
# ---------------------------------------------------------------------------

@display("supplementary:Table S35", stem="S35", title="Reporting conventions and record linkage", scope="protocol")
def table_s35(ctx, out):
    text = protocol_text(ctx, "S35")["rows"]
    values = placeholder_values(ctx)
    problems, rows = [], []
    for key, spec in text.items():
        links = resolved_links(ctx, spec)
        problems += check_linkage(ctx, links, key)
        record = linkage_text(links)
        description = spec.get("record_description")
        if not isinstance(description, str) or not description.strip():
            raise DisplayInputError(f"S35/{key}: the source record description is missing")
        rows.append([key, fill(spec["convention"], values), description, record])
        out.compare(S35_GROUP, (key, "supporting_record"), description)
    raise_problems(problems)
    absent = [key for key, spec in text.items() if spec.get("absent")]
    out.uses("repository", "linkage", ["paper/protocol/S35.json", "records/specs.json", *EXAMPLE_RUNS])
    out.table(["ID", "convention", "supporting record", "repository location"], rows,
              notes=["The supporting-record column preserves the source PDF description. The additional "
                     "repository-location column identifies the checked implementation files.",
                     "Supporting records are artifacts of this repository (run-archive files written by the runtime, "
                     "record sets and the experiment code that produces them, campaign exports, modules and tests); "
                     "every path was checked to exist.",
                     *([f"Conventions whose supporting record the released code does not produce: {', '.join(absent)}; "
                        "a record contract that covers the convention's display is linked where one is declared."]
                       if absent else [])])
    out.data("linkage", ["convention", "kind", "item", "verified"], linkage_rows(ctx, text))


# ---------------------------------------------------------------------------
# Supplementary Table S36: E1–E20 experiment definitions
# ---------------------------------------------------------------------------

def experiment_implementations(ctx) -> dict[str, str]:
    """'Existing implementation or check' of every experiment in docs/PAPER_SUPPORT.md."""
    found = {}
    for line in source_text(ctx, "docs/PAPER_SUPPORT.md").splitlines():
        match = re.match(r"\| \*\*(E\d+) — [^|]*\*\* \| ([^|]*) \|", line)
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


@display("supplementary:Table S36", stem="S36", title="E1–E20 experiment and falsification definitions",
         scope="protocol")
def table_s36(ctx, out):
    text = protocol_text(ctx, "S36")["rows"]
    problems = [p for key, spec in text.items() for p in check_linkage(ctx, {"displays": spec["displays"]}, key)]
    raise_problems(problems)
    cfgs = configs(ctx)
    registered = common([c["registry"]["ensemble_sizes"] for c in cfgs.values()], "ensemble sizes")
    extended = ctx.constants["observer_eq12_table_S9"]["extended_diagnostic_grid_E14"]
    seeds = common([len(c["training"]["seeds"]) for c in cfgs.values()], "training seeds")
    beyond = [e for e in extended if e not in registered]
    values = {"extended_grid": braces(extended), "beyond_registered": " and ".join(map(str, beyond)),
              "plant_seeds": seeds}
    rows = [[key, fill(spec["question"], values), spec["readout"], spec["interpretation"], spec["scope"]]
            for key, spec in text.items()]
    out.compare(S36_GROUP, ("E14", "extended_grid"), extended)
    out.compare(S36_GROUP, ("E14", "distinct_checkpoints_beyond_registry"), beyond)
    out.compare(S36_GROUP, ("E16", "plant_seeds"), seeds)
    implementations = experiment_implementations(ctx)
    out.uses("repository", "experiments", ["paper/protocol/S36.json", "docs/PAPER_SUPPORT.md",
                                           "paper/display_manifest.json"])
    out.table(["ID", "question and controlled comparison", "primary readout", "decision interpretation",
               "reporting scope"], rows,
              notes=["E3–E6 use D_compare only after the teacher and all arm configurations are frozen. The E14 grid "
                     "is the registered extended diagnostic grid; the twelve plant seeds are those of the full "
                     "configurations. Every display named in the reporting scope exists in paper/display_manifest.json."])
    out.data("experiments", ["experiment", "reporting_displays", "repository_implementation"],
             [[key, " ".join(spec["displays"]), implementations.get(key)] for key, spec in text.items()])


# ---------------------------------------------------------------------------
# Supplementary Table S38: results-record schema
# ---------------------------------------------------------------------------

def results_csv_columns(ctx) -> list[str]:
    """Columns of final/results.csv, as _stage_evaluate of cross_morse/pipeline.py writes them."""
    fn = find_function(source_tree(ctx, "cross_morse/pipeline.py"), "_stage_evaluate")
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "fields" for t in node.targets):
            value = literal(node.value)
            if isinstance(value, list) and "model" in value:
                return value
    raise DisplayInputError("cross_morse/pipeline.py writes no results.csv field list")


def evidence_fields(ctx) -> set[str]:
    """Top-level fields of the evidence record of paper/build/evidence.py::evidence_from_run."""
    fn = find_function(source_tree(ctx, "paper/build/evidence.py"), "evidence_from_run")
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "record" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise DisplayInputError("paper/build/evidence.py builds no evidence record")


@display("supplementary:Table S38", stem="S38", title="Results-record schema", scope="protocol")
def table_s38(ctx, out):
    text = protocol_text(ctx, "S38")["rows"]
    key_columns = {k for spec in record_sets(ctx).values() for k in spec.get("keys", [])}
    summary = ctx.record_specs["summary_columns"]
    record_roles = {**{k: "key" for k in key_columns}, **{c: "value" for c in summary[:3]},
                    **{c: "provenance" for c in summary[3:]}}
    csv_columns = results_csv_columns(ctx)
    csv_roles = {c: ("key" if c in ("model", "ensemble") else "value") for c in csv_columns}
    evidence = evidence_fields(ctx)
    problems, rows, mapping = [], [], []
    for key, spec in text.items():
        problems += [f"{key}: records column {c} is not in a record-set schema" for c in spec["records"]
                     if c not in record_roles]
        problems += [f"{key}: results.csv column {c} is not written" for c in spec["results_csv"]
                     if c not in csv_roles]
        problems += [f"{key}: evidence field {c} is not recorded" for c in spec["evidence"] if c not in evidence]
        roles = [record_roles[c] for c in spec["records"] if c in record_roles] + \
                [csv_roles[c] for c in spec["results_csv"] if c in csv_roles]
        role = next((r for r in ("key", "value", "provenance") if r in roles), "not a record column")
        rows.append([spec["field"], spec["values"], role])
        mapping.append([key, " ".join(spec["records"]), " ".join(spec["results_csv"]), " ".join(spec["evidence"]),
                        role])
        out.compare(S38_GROUP, (key, "role"), role)
    raise_problems(problems)
    out.uses("repository", "schemas", ["records/specs.json", "cross_morse/pipeline.py:_stage_evaluate",
                                       "paper/build/evidence.py:evidence_from_run"])
    out.table(["field", "values", "role"], rows,
              notes=["Role: the role of the columns that carry the field in this repository's row-structured records — "
                     f"the measurement-record summary (set keys; {', '.join(summary[:3])}; {', '.join(summary[3:])}) "
                     f"and the run results table final/results.csv ({', '.join(csv_columns)}); a field carried by "
                     "neither is marked 'not a record column'. The per-run evidence record of paper/build/evidence.py "
                     "carries the run-level fields listed in the source data."])
    out.data("field_map", ["field", "record_columns", "results_csv_columns", "evidence_fields", "role"], mapping)


# ---------------------------------------------------------------------------
# Supplementary Table S39: separated data-role contract
# ---------------------------------------------------------------------------

@display("supplementary:Table S39", stem="S39", title="Separated data-role contract", scope="protocol")
def table_s39(ctx, out):
    text = protocol_text(ctx, "S39")["rows"]
    shares = role_shares()
    rule = ctx.source_json("data/sources.json")["role_split_rule"]
    counts = ctx.constants["data_roles_table_S2"]["multi_graph_counts"]
    sizes = {d: role_sizes(row["official_train"]) for d, row in counts.items()}

    def series(values):
        values = [f"{v:,}" for v in values]
        return ", ".join(values[:-1]) + " and " + values[-1]
    values = {"explore_pct": pct(shares["explore"]), "confirm_pct": pct(shares["confirm"]),
              "compare_pct": pct(shares["compare"]), "admit_split": rule["admit"],
              "confirm_counts": series([s["confirm"] for s in sizes.values()]),
              "compare_counts": series([s["compare"] for s in sizes.values()]),
              "admit_counts": series([row["D_admit"] for row in counts.values()])}
    rows = [[spec["stage"], fill(spec["single"], values), fill(spec["multi"], values), fill(spec["reserved"], values)]
            for spec in text.values()]
    for dataset, s in sizes.items():
        out.compare(S39_GROUP, (dataset, "D_confirm"), s["confirm"])
        out.compare(S39_GROUP, (dataset, "D_compare"), s["compare"])
        out.compare(S39_GROUP, (dataset, "D_admit"), counts[dataset]["D_admit"])
    out.compare(S39_GROUP, ("allocation_percent",), [round(100 * v, 6) for v in shares.values()])
    out.uses("code", "cross_morse/split_contract.py", ["TRAIN_CUTS"])
    out.uses("repository", "data/sources.json", ["role_split_rule"])
    out.table(["stage", "single-graph benchmarks (arxiv, proteins, products, Roman)",
               "multigraph benchmarks (Pascal, COCO, Peptides)", "role reserved at preparation"], rows,
              notes=[f"Training allocation {'/'.join(f'{100 * v:.0f}' for v in shares.values())}% from TRAIN_CUTS of "
                     "cross_morse/split_contract.py; multigraph role sizes from the preparation rule int(n · cut) of "
                     "cross_morse/data.py applied to the official training counts "
                     f"({', '.join(f'{d} {row['official_train']:,}' for d, row in counts.items())}; "
                     "paper/registered_constants.json); admission split: data/sources.json."])
    out.data("multigraph_roles", ["dataset", "official_train", "fit", "explore", "select", "confirm", "compare",
                                  "admit"],
             [[d, counts[d]["official_train"], *s.values(), counts[d]["D_admit"]] for d, s in sizes.items()])


# ---------------------------------------------------------------------------
# Supplementary Table S40: comparison-arm information restrictions
# ---------------------------------------------------------------------------

@display("supplementary:Table S40", stem="S40", title="Comparison-arm information restrictions", scope="protocol")
def table_s40(ctx, out):
    text = protocol_text(ctx, "S40")["rows"]
    raise_problems(arm_checks(ctx, text))
    rows = [[spec["arm_label"], spec["allowed"], spec["forbidden"], spec["completion"]] for spec in text.values()]
    for key, spec in text.items():
        for field in ("allowed", "forbidden", "completion"):
            out.compare(S40_GROUP, [key, field], spec[field])
    out.uses("code", "cm_experiments", ["protocol.py:ARMS", "execution.py", "budget.py"])
    out.table(["arm", "inputs allowed", "inputs forbidden", "completion and cost rule"], rows,
              notes=["Teachers and arm configurations are fixed before D_compare; every arm of "
                     "cm_experiments.protocol.ARMS has a row and each restriction names the module that enforces it "
                     "(source data)."])
    out.data("arms", ["row", "arm", "implementation"],
             [[k, s.get("arm"), s["implementation"]] for k, s in text.items()])


# ---------------------------------------------------------------------------
# Main Figure 1: frozen predictor and policy lifecycle
# ---------------------------------------------------------------------------

def lifecycle_content(ctx) -> dict:
    """Box contents of the lifecycle illustration, read from configuration and code."""
    cfgs = configs(ctx)
    actions = action_registry(ctx)
    routes = route_registry(ctx)
    trajectories = [c["exploration"]["trajectories"] for c in cfgs.values()]
    support = common([c["support"] for c in cfgs.values()], "support thresholds")
    controller = common([{k: c["controller"][k] for k in ("state_dim", "covariance_rank")} for c in cfgs.values()],
                        "observer settings")
    selection = common([{k: c["selection"][k] for k in ("outer_folds", "inner_folds", "epsilon_eq")}
                        for c in cfgs.values()], "selection settings")
    admission = common([{k: c["admission"][k] for k in ("clip", "alpha")} for c in cfgs.values()], "admission")
    labels = protocol_text(ctx, "S10")["rows"]
    route_names = {route: labels[route]["route"] for route, _, _ in routes}
    gates = [row["gate"] for row in protocol_text(ctx, "S3")["rows"].values()]
    contrasts = [row["contrast"] for row in protocol_text(ctx, "S11")["rows"].values()][:3]
    recovery = [a.name.replace("_", " ") for a in actions if a.mode != "transport"]
    schema = measurement_schema(ctx, cfgs["ogbn-arxiv"]["sensor"]["bands"], 1)
    sensors = protocol_text(ctx, "S8")["rows"]
    measured = [spec["measurement"] for spec in sensors.values() if spec["kind"] in ("cheap", "spectral")]
    return {
        "trajectories": f"{min(trajectories):,}–{max(trajectories):,} trajectories",
        "actions": f"{len(actions)} actions (A0–A{len(actions) - 1})",
        "support": f"support: p ≥ {support['p_min']:g}, ESS ≥ {support['n_min']}",
        "measurements": measured, "schema_size": len(schema),
        "observer": f"state observer ({controller['state_dim']}-dim, rank {controller['covariance_rank']})",
        "open_loop": [route_names[r] for r, _, ol in routes if ol and r != "nominal"],
        "feedback": [route_names[r] for r, _, ol in routes if not ol],
        "selection": (f"{selection['outer_folds']} outer × {selection['inner_folds']} inner folds; "
                      f"$\\epsilon_{{eq}}$ = {selection['epsilon_eq']:g}"),
        "holm": f"Holm α = {confirmation_alpha():g}",
        "recovery": ["nominal (A0)"] + [f"{name} (A{a.id})" for name, a in
                                        zip(recovery, [a for a in actions if a.mode != "transport"])],
        "gates": gates, "contrasts": contrasts,
        "admission": f"B = {admission['clip']:.2f}, $\\delta_a$ = {admission['alpha']:g}",
    }


@display("main:Figure 1", stem="Fig1", title="Frozen predictor and policy lifecycle", scope="illustration")
def figure_1(ctx, out):
    from matplotlib.patches import FancyBboxPatch, Polygon
    from scripts.draw_diagrams import ART, INK, LINE, MUTED, PLANT, POLICY, SENSE
    content = lifecycle_content(ctx)
    plt = pyplot()
    fig = plt.figure(figsize=(13.0, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 96)
    ax.axis("off")
    nodes, edges = [], []

    def panel(y, h, title):
        ax.add_patch(FancyBboxPatch((1, y), 128, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                    ec=LINE, lw=1.0))
        ax.text(2.5, y + h - 2.2, title, fontsize=10, weight="bold", color=INK, va="center")

    def box(key, x, y, w, h, title, lines=(), color=ART, panel_name=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=color[1],
                                    ec=color[0], lw=1.1))
        wrapped = textwrap.fill(title, max(8, int(w / 0.62)))
        ax.text(x + w / 2, y + h - 1.6, wrapped, fontsize=7.4, weight="bold", color=color[0], ha="center", va="top")
        body = "\n".join(line if line.startswith("$") else
                         textwrap.fill(f"• {line}" if len(lines) > 1 else line, max(8, int(w / 0.52)))
                         for line in lines)
        top = y + h - 2.2 - 1.55 * (wrapped.count("\n") + 1)
        ax.text(x + 0.9 if len(lines) > 1 else x + w / 2, top, body, fontsize=6.4, color=INK,
                ha="left" if len(lines) > 1 else "center", va="top", linespacing=1.25)
        nodes.append([panel_name, key, title, " | ".join(lines)])
        return (x, y, w, h)

    def diamond(key, cx, cy, w, h, title, color=POLICY, panel_name=""):
        ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                             closed=True, fc=color[1], ec=color[0], lw=1.1))
        ax.text(cx, cy, title, fontsize=7, weight="bold", color=color[0], ha="center", va="center")
        nodes.append([panel_name, key, title, ""])

    def arrow(a, b, start, end, label="", color=LINE, dashed=False, via=(), beside=False):
        points = [start, *via]
        for p, q in zip(points, points[1:]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.0, ls="--" if dashed else "-")
        ax.annotate("", xy=end, xytext=points[-1], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                                                   linestyle="--" if dashed else "-",
                                                                   shrinkA=0, shrinkB=0))
        if label:
            x, y = (points[-1][0] + end[0]) / 2, (points[-1][1] + end[1]) / 2
            ax.text(x + 0.8 if beside else x, y if beside else y + 1.0, label, fontsize=6, color=MUTED,
                    ha="left" if beside else "center", va="center" if beside else "bottom")
        edges.append([a, b, label])

    # Panel 1: offline development and qualification
    p1 = "1. Offline development and qualification"
    panel(68, 27, p1)
    y, h = 70, 20
    specs = [
        ("q0", 3, 10, "Frozen graph model $q_0$", [], PLANT),
        ("qN", 15, 11, "Controlled nominal process $q_N^{ctrl}$", [], PLANT),
        ("explore", 28.5, 15, "Exploration and logging", [content["trajectories"], content["actions"], "losses",
                                                          content["support"]], ART),
        ("measure", 46, 17, "State measurements", content["measurements"], SENSE),
        ("models", 65.5, 14, "Observer and risk models", [content["observer"], "one-step / sequence risk predictors"],
         SENSE),
        ("families", 82, 18, "Candidate policy families",
         ["Open-loop: " + ", ".join(content["open_loop"]), "Feedback: " + ", ".join(content["feedback"])], POLICY),
        ("select", 102.5, 12, "Select strongest open-loop comparator $q_C$ and provisional $\\hat{q}_s$",
         [content["selection"]], POLICY),
        ("confirm", 117, 11, "Held-out confirmation: confirmed policy $q_M$", [content["holm"]], POLICY),
    ]
    boxes = {}
    for key, x, w, title, lines, color in specs:
        boxes[key] = box(key, x, y, w, h, title, lines, color, p1)
    order = [s[0] for s in specs]
    for a, b in zip(order, order[1:]):
        xa, ya, wa, ha = boxes[a]
        xb = boxes[b][0]
        arrow(a, b, (xa + wa, ya + ha / 2), (xb, ya + ha / 2))

    # Panel 2: executable deployment logic
    p2 = "2. Executable deployment logic"
    panel(29, 37, p2)
    box("input", 3, 42, 10, 8, "Input graph $x$", [], ART, p2)
    diamond("gate", 21, 46, 12, 10, "Static\ncompatibility\ngate $g_{input}(x)$", POLICY, p2)
    ax.add_patch(FancyBboxPatch((31, 34.5), 52, 29, boxstyle="round,pad=0,rounding_size=0.8", fc="white",
                                ec=POLICY[0], lw=1.2, ls="--"))
    ax.text(57, 61.5, "Executable controller $\\tilde{q}_M$ — internal control loop (ℓ = 0, …, L − 1)", fontsize=7.6,
            weight="bold", color=POLICY[0], ha="center", va="center")
    nodes.append([p2, "controller", "Executable controller q̃_M", "internal control loop"])
    loop = [("state", 33, 9, "state $s_\\ell$"), ("support", 45, 11, "supported action set $A_{sup}(s_\\ell)$"),
            ("policy", 59, 10, "policy $\\pi_q(s_\\ell)$"), ("transport", 72, 10, "controlled transport / update")]
    for key, x, w, title in loop:
        boxes[key] = box(key, x, 45, w, 8, title, [], SENSE if key == "state" else POLICY, p2)
    for (a, xa, wa, _), (b, xb, _, _) in zip(loop, loop[1:]):
        arrow(a, b, (xa + wa, 49), (xb, 49))
    box("recovery", 40, 36, 32, 6.5, "recovery actions", [", ".join(content["recovery"])], ART, p2)
    arrow("support", "recovery", (50.5, 45), (50.5, 42.5), "unsupported / empty set", dashed=True, beside=True)
    arrow("transport", "state", (77, 53), (37.5, 53), "advance to $s_{\\ell+1}$", via=((77, 55.2), (37.5, 55.2)))
    arrow("input", "gate", (13, 46), (15, 46))
    arrow("gate", "controller", (27, 46), (31, 46), "compatible")
    diamond("admit", 92, 46, 12, 10, "Deployment\nadmission\n$G_{admit}$", POLICY, p2)
    arrow("controller", "admit", (83, 46), (86, 46))
    box("deployed", 104, 48.5, 22, 7.5, "Deployed predictor $q_D$", ["$G_{admit}$ = 1 (admit)"], PLANT, p2)
    box("fallback", 104, 35, 22, 8.5, "Exact comparator fallback", ["$G_{admit}$ = 0: returns $q_C$ exactly"],
        ART, p2)
    arrow("admit", "deployed", (98, 46), (104, 52))
    arrow("admit", "fallback", (98, 46), (104, 39))
    box("comparator", 3, 33, 18, 7, "Comparator $q_C$ (strongest open loop)", [], ART, p2)
    arrow("gate", "comparator", (21, 41), (14, 40), "incompatible")

    # Panel 3: evidence and reporting
    p3 = "3. Evidence and reporting"
    panel(5, 21, p3)
    evidence = [
        ("qualification", "Target-regime qualification", content["gates"], ART),
        ("mechanism", "Mechanism tests", content["contrasts"], ART),
        ("admission", "Admission statistics",
         ["$U_a = \\bar{D} + \\sqrt{2 S^2 \\log(2/\\delta_a)/J} + 14 B \\log(2/\\delta_a)/(3(J-1))$",
          content["admission"]], ART),
        ("cost", "Ablation and cost", ["factorial", "within-block ablations", "expected / worst-case cost"], ART),
        ("outputs", "Final outputs", ["official metric", "clipped loss", "calibration", "fallback rate"], ART),
    ]
    x = 3
    for key, title, lines, color in evidence:
        w = 30 if key == "admission" else 21
        box(key, x, 7, w, 14, title, lines, color, p3)
        x += w + 2.5

    # Legend
    legend = "  ·  ".join(["$q_0$ = upstream predictor (frozen graph model)", "$q_N^{ctrl}$ = controlled nominal process",
                           "$q_C$ = strongest open-loop comparator", "$q_M$ = confirmed policy",
                           "$q_D$ = deployed rule"])
    ax.text(65, 2.2, legend, fontsize=6.8, color=INK, ha="center", va="center")
    out.figure(fig)
    out.data("nodes", ["panel", "node", "title", "content"], nodes)
    out.data("edges", ["source", "target", "label"], edges)
    out.uses("configuration", "configs/*_full.json", ["exploration", "support", "controller", "selection",
                                                      "admission"])
    out.uses("code", "cross_morse", ["core.py:ACTIONS", "core.py:MeasurementBank", "policies.py:make_registry",
                                     "statistics.py:confirmation"])
