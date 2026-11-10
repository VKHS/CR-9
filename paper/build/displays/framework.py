"""Display generators: one function per numbered table and figure of the paper.

A generator reads experiment artifacts (sealed lifecycle runs, verified
measurement-record sets and campaign exports) together with the repository's own
configurations, registered protocol constants and analytical code. It writes the
complete display: a table as CSV, Markdown and LaTeX, a figure as PNG and SVG with
one CSV of plotted source data per panel.

The reported-results registry (``paper/reported_results.json``) is never read by
a generator. Generators declare the printed value that each computed cell should
reproduce; :func:`compare` looks the printed value up afterwards and records
whether the two agree at the printed precision.
"""
from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire",
            "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "Roman", "PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}
NODE_LEVEL = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire")
GRAPH_LEVEL = ("PascalVOC-SP", "COCO-SP", "Peptides-func")
MISSING_TEXT = "—"


class DisplayInputError(ValueError):
    """A declared input of a display is absent or does not hold the required field."""


class DisplayValidationError(ValueError):
    """Supplied evidence is inconsistent or invalid, rather than merely absent."""


@dataclasses.dataclass(frozen=True)
class NotApplicable:
    reason: str


def not_applicable(reason: str) -> NotApplicable:
    """An intentional nonnumeric cell, distinct from a missing observation."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A not-applicable cell requires its reason")
    return NotApplicable(reason)


# ---------------------------------------------------------------------------
# Input views
# ---------------------------------------------------------------------------

class RunView:
    """Read access to one sealed lifecycle run, or to a Roman-empire split campaign."""

    def __init__(self, dataset: str, root: Path, *, splits: Iterable["RunView"] = (), evidence: dict | None = None):
        self.dataset = dataset
        self.root = Path(root)
        self.splits = list(splits)
        self._evidence = evidence

    @property
    def is_campaign(self) -> bool:
        return bool(self.splits)

    def members(self) -> list["RunView"]:
        """The individual runs: the ten splits of a campaign, otherwise the run itself."""
        return self.splits if self.splits else [self]

    @functools.cached_property
    def evidence(self) -> dict:
        if self._evidence is not None:
            return self._evidence
        from paper.build.evidence import evidence_from_campaign, evidence_from_run
        return evidence_from_campaign(self.root) if self.is_campaign else evidence_from_run(self.root)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def exists(self, relative: str) -> bool:
        return (self.root / relative).is_file()

    def json(self, relative: str, default: Any = ...) -> Any:
        path = self.root / relative
        if not path.is_file():
            if default is ...:
                raise DisplayInputError(f"{self.dataset}: {relative} is absent from {self.root}")
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def npz(self, relative: str) -> dict[str, np.ndarray]:
        path = self.root / relative
        if not path.is_file():
            raise DisplayInputError(f"{self.dataset}: {relative} is absent from {self.root}")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def csv_rows(self, relative: str) -> list[dict[str, str]]:
        path = self.root / relative
        if not path.is_file():
            raise DisplayInputError(f"{self.dataset}: {relative} is absent from {self.root}")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def glob(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern))


class RecordSetView:
    """A verified measurement-record set: summary rows traced to raw files."""

    def __init__(self, name: str, root: Path | None, spec: dict, rows: list[dict], *, synthetic: bool = False):
        self.name = name
        self.root = Path(root) if root is not None else None
        self.spec = spec
        self.rows = rows
        self.synthetic = synthetic

    @property
    def keys(self) -> list[str]:
        return list(self.spec.get("keys", []))

    def select(self, quantity: str | None = None, **keys) -> list[dict]:
        out = []
        for row in self.rows:
            if quantity is not None and row.get("quantity") != quantity:
                continue
            if any(str(row.get(k)) != str(v) for k, v in keys.items()):
                continue
            out.append(row)
        return out

    def value(self, quantity: str, **keys) -> float | None:
        rows = self.select(quantity, **keys)
        if not rows:
            return None
        if len(rows) > 1:
            raise DisplayInputError(f"{self.name}: {quantity} {keys} is not unique ({len(rows)} rows)")
        return rows[0]["value"]

    def require(self, quantity: str, **keys) -> float:
        value = self.value(quantity, **keys)
        if value is None:
            raise DisplayInputError(f"{self.name}: {quantity} {keys} is not in the record set")
        return value

    def distinct(self, key: str, quantity: str | None = None) -> list[str]:
        seen = []
        for row in self.select(quantity):
            value = row.get(key)
            if value not in seen and value not in (None, ""):
                seen.append(value)
        return seen

    def raw_path(self, relative: str) -> Path:
        if self.root is None:
            raise DisplayInputError(f"{self.name}: raw files are not available")
        return self.root / relative


class DisplayContext:
    """Everything a generator may read. The paper registry is deliberately absent."""

    def __init__(self, *, root: Path = ROOT, runs: dict[str, RunView] | None = None,
                 records: dict[str, RecordSetView] | None = None,
                 campaigns: dict[str, Path] | None = None, synthetic: bool = False, demonstration: bool = False):
        self.root = Path(root)
        self._runs = dict(runs or {})
        self._records = dict(records or {})
        self._campaigns = {k: Path(v) for k, v in (campaigns or {}).items()}
        self.synthetic = synthetic
        # the CPU demonstration (scripts/reconstruct_paper.py --demo): its campaigns ran on fixture parents
        self.demonstration = demonstration

    # -- experiment artifacts ------------------------------------------------
    def run(self, dataset: str) -> RunView | None:
        return self._runs.get(dataset)

    def require_run(self, dataset: str) -> RunView:
        view = self.run(dataset)
        if view is None:
            raise DisplayInputError(f"the {dataset} lifecycle run is required")
        return view

    def runs(self) -> dict[str, RunView]:
        return {d: self._runs[d] for d in DATASETS if d in self._runs} | {
            d: v for d, v in self._runs.items() if d not in DATASETS}

    def records(self, name: str) -> RecordSetView | None:
        return self._records.get(name)

    def require_records(self, name: str) -> RecordSetView:
        view = self.records(name)
        if view is None:
            raise DisplayInputError(f"the {name} measurement-record set is required")
        return view

    def campaign(self, name: str) -> Path | None:
        return self._campaigns.get(name)

    def require_campaign(self, name: str) -> Path:
        path = self.campaign(name)
        if path is None:
            raise DisplayInputError(f"the {name} campaign export is required")
        return path

    def available(self, need: str) -> bool:
        kind, _, name = need.partition(":")
        if kind == "static":
            return True
        if kind == "runs":
            return all(d in self._runs for d in DATASETS)
        if kind == "run":
            return name in self._runs
        if kind == "records":
            return name in self._records
        if kind == "campaign":
            return name in self._campaigns
        raise ValueError(f"Unknown display input declaration: {need}")

    # -- repository sources --------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def config(self, dataset: str, profile: str = "full") -> dict:
        path = self.root / "configs" / f"{STEMS[dataset]}_{profile}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @functools.cached_property
    def constants(self) -> dict:
        return json.loads((self.root / "paper/registered_constants.json").read_text(encoding="utf-8"))

    @functools.cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "paper/display_manifest.json").read_text(encoding="utf-8"))

    @functools.cached_property
    def record_specs(self) -> dict:
        return json.loads((self.root / "records/specs.json").read_text(encoding="utf-8"))

    def source_json(self, relative: str) -> Any:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def num(value, decimals: int = 4, *, signed: bool = False, suffix: str = "") -> str | None:
    """Fixed-point text at the printed precision; ``None`` stays missing."""
    number = finite(value)
    if number is None:
        return None
    text = f"{number:+.{decimals}f}" if signed else f"{number:.{decimals}f}"
    if text in {"-0." + "0" * decimals, "+0." + "0" * decimals} and not signed:
        text = text.lstrip("+-")
    return text + suffix


def integer(value) -> str | None:
    number = finite(value)
    return None if number is None else str(int(round(number)))


def interval(low, high, decimals: int = 4, *, signed: bool = False) -> str | None:
    lo, hi = num(low, decimals, signed=signed), num(high, decimals, signed=signed)
    return None if lo is None or hi is None else f"[{lo}, {hi}]"


def percent(value, decimals: int = 1) -> str | None:
    number = finite(value)
    return None if number is None else f"{100.0 * number:.{decimals}f}%"


def _text(value) -> str:
    if isinstance(value, NotApplicable):
        return MISSING_TEXT
    if value is None:
        return MISSING_TEXT
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _tex(text: str) -> str:
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c) for c in text)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Comparison:
    group: str
    path: tuple
    computed: Any
    note: str = ""
    missing: bool = False


class DisplayWriter:
    """Collects the files and declared comparisons of one display."""

    def __init__(self, display_id: str, stem: str, out_dir: Path, *, title: str, synthetic: bool,
                 scope: str = "measurement"):
        self.display_id = display_id
        self.stem = stem
        self.out_dir = Path(out_dir)
        self.title = title
        self.synthetic = synthetic
        self.scope = scope
        self.files: list[str] = []
        self.comparisons: list[Comparison] = []
        self.notes: list[str] = []
        self.inputs: list[dict] = []
        self.missing_content: list[dict] = []
        self.not_applicable_cells: list[dict] = []
        self.table_count = 0
        self.figure_count = 0
        self.data_count = 0
        self.evidence_scope: list[dict] = []

    def incomplete(self, reason: str, *, location: str = "display") -> None:
        item = {"location": location, "reason": reason}
        if item not in self.missing_content:
            self.missing_content.append(item)

    def provenance(self, **details) -> None:
        """Carry source-method/history limits into the machine-readable display record."""
        self.evidence_scope.append(details)

    def _write(self, relative: str, text: str) -> str:
        path = self.out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if relative not in self.files:
            self.files.append(relative)
        return relative

    def _banner(self) -> str:
        if self.scope == "protocol":
            scope = "generated from the repository's configuration, code and registered protocol"
        elif self.scope == "analytical":
            scope = "computed from the stated formulas and registered inputs"
        elif self.scope == "illustration":
            scope = "illustration generated from the repository's code and configuration"
        elif self.synthetic:
            scope = "synthetic software example; not the paper's benchmark observations"
        else:
            scope = "computed from the supplied experiment artifacts"
        return f"{self.display_id} — {self.title} ({scope})"

    def table(self, columns: list[str], rows: list[list], *, name: str | None = None,
              notes: Iterable[str] = (), required: bool = True) -> list[str]:
        """Write the complete table. Cells are text (already at printed precision), numbers or None."""
        stem = name or self.stem
        width = len(columns)
        self.table_count += 1
        if required and not rows:
            self.incomplete("The required table has no rows", location=f"tables/{stem}")
        for index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"{self.display_id}: row has {len(row)} cells, header has {width}")
            for column, cell in zip(columns, row):
                location = f"tables/{stem}/row/{index}/{column}"
                if isinstance(cell, NotApplicable):
                    self.not_applicable_cells.append({"location": location, "reason": cell.reason})
                elif required and (cell is None or isinstance(cell, str) and not cell.strip()
                                   or isinstance(cell, (float, np.floating)) and not math.isfinite(cell)):
                    self.incomplete("Required table cell has no finite supplied value", location=location)
        texts = [[_text(cell) for cell in row] for row in rows]
        notes = [*notes, *self.notes]
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([["" if t == MISSING_TEXT else t for t in row] for row in texts])
        files = [self._write(f"tables/{stem}.csv", buffer.getvalue())]
        md = [f"# {self._banner()}", "", "| " + " | ".join(columns) + " |",
              "| " + " | ".join("---" for _ in columns) + " |"]
        md += ["| " + " | ".join(t.replace("|", r"\|").replace("\n", " ") for t in row) + " |" for row in texts]
        md += [""] + [f"{note}" for note in notes]
        files.append(self._write(f"tables/{stem}.md", "\n".join(md).rstrip() + "\n"))
        tex = [f"% {self._banner()}", "\\begin{tabular}{" + "l" * width + "}", "\\hline",
               " & ".join(_tex(c) for c in columns) + r" \\", "\\hline"]
        tex += [" & ".join("--" if t == MISSING_TEXT else _tex(t) for t in row) + r" \\" for row in texts]
        tex += ["\\hline", "\\end{tabular}"] + [f"% {note}" for note in notes]
        files.append(self._write(f"tables/{stem}.tex", "\n".join(tex) + "\n"))
        return files

    def data(self, name: str, columns: list[str], rows: Iterable[Iterable]) -> str:
        """Full-precision source data (CSV) behind a table or a figure panel."""
        rows = list(rows)
        self.data_count += 1
        if not rows:
            self.incomplete("The declared source-data series is empty", location=f"data/{self.stem}/{name}")
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            if len(row) != len(columns):
                raise ValueError(f"{self.display_id}/{name}: source row width differs from its columns")
            if any(isinstance(value, (float, np.floating)) and not math.isfinite(value) for value in row):
                self.incomplete("Source data contain a nonfinite numeric value", location=f"data/{self.stem}/{name}")
            writer.writerow(["" if v is None else (repr(float(v)) if isinstance(v, (float, np.floating))
                                                    else str(v)) for v in row])
        return self._write(f"data/{self.stem}/{name}.csv", buffer.getvalue())

    def figure(self, fig, *, name: str | None = None) -> list[str]:
        """Save a Matplotlib figure as PNG and SVG; panel data go through :meth:`data`."""
        stem = name or self.stem
        self.figure_count += 1
        files = []
        for suffix, options in (("png", {"dpi": 200}), ("svg", {})):
            relative = f"figures/{stem}.{suffix}"
            path = self.out_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", metadata={"Date": None} if suffix == "svg" else {}, **options)
            self.files.append(relative)
            files.append(relative)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return files

    def compare(self, group: str, path: Iterable, computed, *, note: str = "") -> None:
        """Declare that ``computed`` should reproduce ``registry[group][path...]``."""
        self.comparisons.append(Comparison(group, tuple(path), computed, note))

    def compare_missing(self, group: str, path: Iterable, *, reason: str) -> None:
        """Keep an omitted required value in the comparison denominator."""
        self.comparisons.append(Comparison(group, tuple(path), None, reason, missing=True))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def uses(self, kind: str, name: str, files: Iterable[str] = ()) -> None:
        """Record which artifact files the display read (for the source-data trail)."""
        self.inputs.append({"kind": kind, "name": name, "files": sorted(set(files))})


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "svg.hashsalt": "cross-morse", "figure.dpi": 100})
    return plt


# ---------------------------------------------------------------------------
# Registry of generators
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Generator:
    display_id: str
    stem: str
    title: str
    needs: tuple[str, ...]
    function: Callable[[DisplayContext, DisplayWriter], None]
    scope: str


GENERATORS: dict[str, Generator] = {}
SCOPES = {"measurement", "protocol", "analytical", "illustration"}


def display(display_id: str, *, stem: str, title: str, needs: Iterable[str] = ("static",),
            scope: str = "measurement"):
    """Register the generator of one numbered display.

    ``needs`` lists the inputs: ``runs`` (all seven lifecycle runs), ``run:<dataset>``,
    ``records:<set>``, ``campaign:<name>`` or ``static`` (repository sources only).
    ``scope`` says what the display reports: measured observations, protocol
    definitions taken from the configuration, analytical calculations, or an illustration.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown display scope: {scope}")

    def register(function):
        if display_id in GENERATORS:
            raise ValueError(f"Two generators claim {display_id}")
        GENERATORS[display_id] = Generator(display_id, stem, title, tuple(needs), function, scope)
        return function
    return register


# ---------------------------------------------------------------------------
# Comparison with the printed values
# ---------------------------------------------------------------------------

PRINTED_SCHEMA = "cross-morse-printed-display-values-1"
COMPARISON_STATES = ("exact", "differs", "not_comparable", "no_reference", "missing")


def _typed_reference(value):
    return isinstance(value, dict) and "$value" in value


def _validate_reference_values(value):
    if _typed_reference(value):
        if set(value) != {"$value", "$decimals"} or finite(value["$value"]) is None:
            raise DisplayValidationError("A typed printed reference requires a finite $value and $decimals")
        if type(value["$decimals"]) is not int or not 0 <= value["$decimals"] <= 18:
            raise DisplayValidationError("Printed decimal precision must be an integer from 0 to 18")
    elif isinstance(value, dict):
        for child in value.values():
            _validate_reference_values(child)
    elif isinstance(value, list):
        for child in value:
            _validate_reference_values(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise DisplayValidationError("Printed references cannot contain nonfinite numbers")


def load_references(root: Path, registry: dict | None = None) -> tuple[dict, dict, list[dict]]:
    """Read independently reviewed targets only on the comparison side.

    The protocol templates used to render descriptions are deliberately not a
    fallback target: substituting their runtime placeholders would compare the
    implementation with itself. Explicit registry arguments remain useful for
    isolated software tests and do not implicitly load another target file.
    """
    root = Path(root)
    if registry is not None:
        return dict(registry), {}, []
    from cm_campaigns.common import loads, read_json
    base = root / "paper/reported_results.json"
    base_bytes = base.read_bytes()
    values = loads(base_bytes.decode("utf-8"))
    files = [{"path": "paper/reported_results.json", "sha256": hashlib.sha256(base_bytes).hexdigest()}]
    path = root / "paper/printed_display_values.json"
    metadata = {}
    if not path.is_file():
        return values, metadata, files
    sidecar_bytes = path.read_bytes()
    sidecar = loads(sidecar_bytes.decode("utf-8"))
    sources = read_json(root / "paper/sources.json")["documents"]
    manifest = read_json(root / "paper/display_manifest.json")
    displays = {row["id"]: row for row in manifest["displays"]}
    if sidecar.get("schema") != PRINTED_SCHEMA or sidecar.get("documents") != {k: v["sha256"] for k, v in sources.items()}:
        raise DisplayValidationError("Printed display references must identify the current registered PDFs")
    groups = sidecar.get("groups")
    if not isinstance(groups, dict):
        raise DisplayValidationError("Printed display reference groups must be an object")
    for group, entry in groups.items():
        if group in values:
            raise DisplayValidationError(f"Ambiguous duplicate printed reference group: {group}")
        if not isinstance(entry, dict) or entry.get("display_id") not in displays or "values" not in entry:
            raise DisplayValidationError(f"Invalid printed reference group: {group}")
        source = entry.get("source") or {}
        display_row = displays[entry["display_id"]]
        pages = source.get("pages")
        if (source.get("document") != display_row["source"] or not isinstance(pages, list) or not pages
                or any(type(page) is not int or page not in display_row["pages"] for page in pages)):
            raise DisplayValidationError(f"Printed reference page/display identity differs: {group}")
        _validate_reference_values(entry["values"])
        values[group] = entry["values"]
        metadata[group] = {"display_id": entry["display_id"], "source": source}
    files.append({"path": "paper/printed_display_values.json", "sha256": hashlib.sha256(sidecar_bytes).hexdigest()})
    return values, metadata, files

def _decimals(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return 0
    text = repr(value)
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e")
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return max(0, digits - int(exponent))
    return len(text.split(".")[1]) if "." in text else 0


def _lookup(registry: dict, group: str, path: tuple):
    node = registry.get(group)
    if node is None:
        return ..., f"registry group {group} is absent"
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
            node = node[key]
        else:
            return ..., f"{group}/{'/'.join(map(str, path))} is not transcribed"
    return node, ""


def _normal_text(value: str) -> str:
    return " ".join(str(value).replace("−", "-").split()).lower()


def compare_value(computed, reported) -> str:
    """``exact`` at the printed precision, ``differs``, or ``not_comparable``."""
    if _typed_reference(reported):
        number = finite(computed)
        if number is None:
            return "differs" if computed is None else "not_comparable"
        tolerance = 0.5 * 10.0 ** (-reported["$decimals"])
        return "exact" if abs(number - reported["$value"]) <= tolerance * (1 + 1e-9) + 1e-15 else "differs"
    if reported is None:
        return "exact" if computed is None else "differs"
    if isinstance(reported, list):
        if not isinstance(computed, (list, tuple)) or len(computed) != len(reported):
            return "not_comparable"
        states = [compare_value(c, r) for c, r in zip(computed, reported)]
        return "exact" if all(s == "exact" for s in states) else (
            "not_comparable" if "not_comparable" in states else "differs")
    if isinstance(reported, str):
        if computed is None:
            return "differs"
        left = re.sub(r"([<>]=?)\s+", r"\1", _normal_text(computed))
        right = re.sub(r"([<>]=?)\s+", r"\1", _normal_text(reported))
        return "exact" if left == right else "differs"
    if isinstance(reported, bool):
        return "exact" if computed is reported else "differs"
    number = finite(computed)
    if number is None:
        return "differs" if computed is None else "not_comparable"
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        return "not_comparable"
    places = _decimals(reported)
    tolerance = 0.5 * 10.0 ** (-places) if places is not None else 0.0
    return "exact" if abs(number - float(reported)) <= tolerance * (1 + 1e-9) + 1e-15 else "differs"


def compare(comparisons: list[Comparison], registry: dict) -> list[dict]:
    rows = []
    for item in comparisons:
        reported, reason = _lookup(registry, item.group, item.path)
        if reported is ...:
            rows.append({"group": item.group, "path": "/".join(map(str, item.path)), "computed": item.computed,
                         "reported": None, "status": "no_reference", "note": reason})
            continue
        missing = item.missing or item.computed is None and reported is not None
        rows.append({"group": item.group, "path": "/".join(map(str, item.path)), "computed": item.computed,
                     "reported": reported, "status": "missing" if missing else compare_value(item.computed, reported),
                     "note": item.note})
    return rows


def _reference_leaves(value, path=(), *, include_text=False):
    if _typed_reference(value):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            if not str(key).startswith("_"):
                yield from _reference_leaves(child, (*path, key), include_text=include_text)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _reference_leaves(child, (*path, index), include_text=include_text)
    elif (value is None or isinstance(value, (int, float, bool)) or include_text
          or isinstance(value, str) and re.fullmatch(r"\s*[<>]=?\s*[0-9.eE+-]+\s*", value)):
        yield path, value


def comparison_coverage(display_id: str, comparisons: list[Comparison], registry: dict, metadata: dict,
                        *, scope: str) -> tuple[list[dict], dict]:
    """Account for required targets independently of which comparisons ran.

    Legacy table groups have independently registered numerical cells. Figures
    need a dedicated reviewed target map: a few related table checks cannot
    establish that every printed panel value was compared.
    """
    role, label = display_id.split(":", 1)
    token = label.replace("Table ", "")
    prefix = (f"main_table_{token}_" if role == "main" else f"supplementary_table_{token}_")
    groups = {name for name in registry if label.startswith("Table ") and name.startswith(prefix)}
    groups.update(name for name, entry in metadata.items() if entry["display_id"] == display_id)
    targets = {(group, path): value for group in groups
               for path, value in _reference_leaves(registry[group], include_text=group in metadata)}
    covered = set()
    for item in comparisons:
        value, reason = _lookup(registry, item.group, item.path)
        if not reason:
            covered.update((item.group, path) for path, _ in _reference_leaves(value, item.path, include_text=True))
    absent = sorted(set(targets) - covered, key=lambda pair: (pair[0], tuple(map(str, pair[1]))))
    extra = [{"group": group, "path": "/".join(map(str, path)), "computed": None,
              "reported": targets[group, path], "status": "missing",
              "note": "No comparison was supplied for this registered printed value"} for group, path in absent]
    if not targets and scope != "illustration":
        extra.append({"group": f"display:{display_id}", "path": "", "computed": None, "reported": None,
                      "status": "missing", "note": "No dedicated independently reviewed printed-value contract is registered for this display"})
    complete = not extra
    return extra, {"expected_values": len(targets), "covered_values": len(set(targets) & covered),
                   "missing_values": len(extra), "complete": complete,
                   "scope": "Registered table numerical cells and reviewed display-specific targets; related-table checks alone do not certify a figure"}


# ---------------------------------------------------------------------------
# Build every registered display
# ---------------------------------------------------------------------------

LOAD_ERRORS: dict[str, str] = {}


def _load_generators() -> None:
    """Import every generator module; a module that fails to import is reported, not skipped silently."""
    import importlib
    import pkgutil
    package = Path(__file__).resolve().parent
    for module in pkgutil.iter_modules([str(package)]):
        if module.name in {"framework", "testing", "__init__"}:
            continue
        try:
            importlib.import_module(f"paper.build.displays.{module.name}")
            LOAD_ERRORS.pop(module.name, None)
        except Exception as exc:  # reported by build_displays and by the statement check
            LOAD_ERRORS[module.name] = f"{type(exc).__name__}: {exc}"


def build_displays(ctx: DisplayContext, out_dir: Path, *, registry: dict | None = None,
                   only: Iterable[str] | None = None) -> dict:
    """Run every generator whose inputs are present; record the rest as awaiting inputs."""
    _load_generators()
    out_dir = Path(out_dir)
    registry, reference_metadata, reference_files = load_references(ctx.root, registry)
    order = [row["id"] for row in ctx.manifest["displays"]]
    wanted = set(only) if only is not None else None
    results = []
    for display_id in order:
        generator = GENERATORS.get(display_id)
        if wanted is not None and display_id not in wanted:
            continue
        if generator is None:
            results.append({"id": display_id, "status": "no_generator"})
            continue
        missing = [need for need in generator.needs if not ctx.available(need)]
        entry = {"id": display_id, "stem": generator.stem, "title": generator.title,
                 "scope": generator.scope, "needs": list(generator.needs)}
        if missing:
            results.append({**entry, "status": "awaiting_inputs", "missing": missing})
            continue
        writer = DisplayWriter(display_id, generator.stem, out_dir, title=generator.title,
                               synthetic=ctx.synthetic, scope=generator.scope)
        try:
            generator.function(ctx, writer)
        except DisplayInputError as exc:
            results.append({**entry, "status": "awaiting_inputs", "missing": [str(exc)]})
            continue
        except Exception as exc:  # a generator defect must not hide the other displays
            import traceback
            results.append({**entry, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=6)})
            continue
        if not writer.files:
            writer.incomplete("The generator wrote no files")
        if ":Table " in display_id and not writer.table_count:
            writer.incomplete("The numbered table was not emitted")
        if ":Figure " in display_id:
            if not writer.figure_count:
                writer.incomplete("The numbered figure was not emitted")
            if generator.scope != "illustration" and not writer.data_count:
                writer.incomplete("The quantitative figure has no plotted source data")
        checks = compare(writer.comparisons, registry)
        missing_checks, coverage = comparison_coverage(display_id, writer.comparisons, registry,
                                                      reference_metadata, scope=generator.scope)
        checks.extend(missing_checks)
        if checks:
            rel = f"compare/{generator.stem}.csv"
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                w = csv.DictWriter(handle, fieldnames=["group", "path", "computed", "reported", "status", "note"],
                                   lineterminator="\n")
                w.writeheader()
                for row in checks:
                    w.writerow({k: json.dumps(v) if isinstance(v, (list, tuple, dict)) else
                                ("" if v is None else v) for k, v in row.items()})
            writer.files.append(rel)
        counts = {s: sum(1 for c in checks if c["status"] == s) for s in COMPARISON_STATES}
        comparison_complete = coverage["complete"] and not any(counts[s] for s in
                                                                ("not_comparable", "no_reference", "missing"))
        coverage["complete"] = comparison_complete
        results.append({**entry, "status": "built", "files": list(writer.files), "comparison": counts,
                        "content_complete": not writer.missing_content,
                        "missing_content": writer.missing_content,
                        "not_applicable_cells": writer.not_applicable_cells,
                        "comparison_complete": comparison_complete,
                        "comparison_coverage": coverage,
                        "evidence_scope": writer.evidence_scope,
                        "inputs": writer.inputs, "synthetic": ctx.synthetic})
    return {"schema": "cross-morse-displays-1", "displays": results,
            "generators": len([r for r in results if r["status"] != "no_generator"]),
            "built": len([r for r in results if r["status"] == "built"]),
            "errors": [r["id"] for r in results if r["status"] == "error"],
            "module_errors": dict(LOAD_ERRORS), "comparison_sources": reference_files}
