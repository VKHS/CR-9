"""Validate the published views used by the offline availability check.

Existence and a self-reported completion count are not evidence of a published
experiment. These checks bind the small committed views to the release manifest,
the actual study contracts and the reconstruction's file inventory. Download and
execution checks remain the responsibility of ``make paper``.
"""
from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def inside_file(root, relative):
    """Resolve an ordinary, relative file without following a symlink."""
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "view file names must be nonempty relative POSIX paths")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts and parts.as_posix() == relative,
            f"view file is outside its directory or has a noncanonical name: {relative}")
    path = root
    for part in parts.parts:
        path = path / part
        require(not path.is_symlink(), f"view file follows a symlink: {relative}")
    require(path.is_file() and path.resolve().is_relative_to(root), f"view file is missing: {relative}")
    return path


def check_digest(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"{path}: no valid SHA-256 identity")
    require(sha256(path) == expected, f"{path}: differs from its published identity")


def check_snapshot(root, folder, manifest, *, exact_files=True, check_builders=False):
    """Validate a derived INDEX and all files that it claims to represent."""
    from scripts import paper_artifacts as pa
    folder = Path(folder)
    index = read_json(inside_file(folder, "INDEX.json"))
    check_digest(Path(root) / "release/paper_artifacts.json", index.get("artifact_manifest_sha256"))
    files = index.get("files")
    require(isinstance(files, dict) and files, f"{folder}: empty or absent file inventory")
    require("INDEX.json" not in files, f"{folder}: INDEX cannot include itself")
    for name, digest in files.items():
        check_digest(inside_file(folder, name), digest)
    if exact_files:
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() or p.is_symlink()} - {"INDEX.json"}
        require(actual == set(files), f"{folder}: file inventory omits or adds files: "
                + ", ".join(sorted(actual ^ set(files))))
    links = index.get("artifacts")
    require(isinstance(links, list), f"{folder}: no published artifact identities")
    published = {a["id"]: a for a in manifest["artifacts"] if a.get("status") == pa.PUBLISHED}
    seen = set()
    for link in links:
        require(isinstance(link, dict), f"{folder}: invalid artifact identity")
        ident = link.get("id")
        require(ident in published and ident not in seen, f"{folder}: stale or duplicate artifact {ident}")
        seen.add(ident)
        entry = published[ident]
        tier = entry.get("tiers", {}).get(link.get("tier"))
        require(tier and link.get("archive_sha256") == tier.get("sha256"),
                f"{folder}: {ident} archive differs from the current release")
        for key in ("kind", "relationship", "inventory_sha256", "identity"):
            require(link.get(key) == entry.get(key), f"{folder}: {ident} {key} differs from the current release")
        require(link.get("scientific_only_verification") is not False
                and entry.get("scientific_only_verification") is not False,
                f"{folder}: {ident} was packed only as a software fixture")
    require(seen == set(published), f"{folder}: artifact identities do not cover the published release")
    if check_builders:
        expected = pa.builder_source_inventory(Path(root))
        require(index.get("builder_sources") == expected,
                f"{folder}: reconstruction tools or their registered inputs changed; rebuild the published views")
    return index


def check_baseline_studies(view_path):
    """Validate the fourteen actual E12 studies, not just fourteen dictionaries."""
    from cm_campaigns.contract import EXTERNAL
    from cm_experiments.upstream import validate_plan
    from scripts.paper_artifacts import BASELINE_SCHEMA
    view_path = Path(view_path)
    view = read_json(view_path)
    require(view.get("schema") == BASELINE_SCHEMA and view.get("source") == "records:baselines",
            "baseline study view has no valid schema/source")
    entries = view.get("studies")
    require(isinstance(entries, list), "baseline study view requires a studies list")
    expected = {(dataset, method) for dataset, methods in EXTERNAL.items() for method in methods}
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict), "baseline study entry must be an object")
        pair = (entry.get("dataset"), entry.get("method"))
        require(pair in expected and pair not in seen, f"baseline view has an unexpected or duplicate study: {pair}")
        seen.add(pair)
        paths = {}
        for key in ("plan", "upstream", "adapter_archive", "study_manifest"):
            descriptor = entry.get(key)
            require(isinstance(descriptor, dict), f"{pair}: {key} file and SHA-256 are required")
            path = inside_file(view_path.parent, descriptor.get("file"))
            check_digest(path, descriptor.get("sha256"))
            paths[key] = path
        plan = validate_plan(read_json(paths["plan"]))
        upstream = read_json(paths["upstream"])
        require((plan["dataset"], plan["method"]) == pair
                and (upstream.get("dataset"), upstream.get("method")) == pair,
                f"{pair}: study identity differs between its plan and upstream record")
        for key in ("source_commit", "adapter_commit"):
            require(entry.get(key) == plan[key] == upstream.get(key), f"{pair}: {key} does not match the study plan")
        archive = upstream.get("adapter_archive") or {}
        require(archive.get("commit") == plan["adapter_commit"]
                and archive.get("sha256") == entry["adapter_archive"]["sha256"],
                f"{pair}: adapter archive differs from its pinned upstream record")
        try:
            with tarfile.open(paths["adapter_archive"], "r:*") as handle:
                regular = [member for member in handle.getmembers() if member.isfile() and member.size > 0]
                require(regular, f"{pair}: adapter archive contains no source files")
                for member in handle.getmembers():
                    path = PurePosixPath(member.name)
                    require(not path.is_absolute() and ".." not in path.parts,
                            f"{pair}: adapter archive contains an unsafe file name")
                require(handle.pax_headers.get("comment") == plan["adapter_commit"],
                        f"{pair}: adapter archive is not bound to the pinned git commit")
        except tarfile.TarError as exc:
            raise ValueError(f"{pair}: adapter archive is unreadable: {exc}") from exc
        _check_baseline_outputs(entry, plan, upstream, read_json(paths["study_manifest"]))
    require(seen == expected, "baseline view omits registered studies: "
            + ", ".join(f"{dataset}/{method}" for dataset, method in sorted(expected - seen)))
    return view


def _check_baseline_outputs(entry, plan, upstream, inventory):
    """Check the small study inventory; full archive bytes are checked by the complete-tier gate."""
    from cm_campaigns.baselines import STUDY_ARTIFACT_SCHEMA, study_paths
    from cm_experiments.artifacts import canonical
    where = f"{plan['dataset']}/{plan['method']}"
    archive = entry.get("study_archive")
    original = upstream.get("study_archive")
    require(isinstance(archive, dict) and isinstance(original, dict),
            f"{where}: complete original study archive is not linked")
    require(archive.get("artifact") == "records:baselines"
            and all(original.get(key) == archive.get(key) for key in ("file", "sha256", "bytes", "manifest_file", "manifest_sha256"))
            and all(archive.get(key) == value for key, value in original.items()),
            f"{where}: study archive descriptor differs from the released record set")
    paths = study_paths(plan["dataset"], plan["method"], plan["adapter_commit"])
    require(archive.get("file") == paths["study_archive"]
            and archive.get("manifest_file") == paths["study_manifest"],
            f"{where}: study archive has no canonical release location")
    require(type(archive.get("bytes")) is int and archive["bytes"] > 0
            and isinstance(archive.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]),
            f"{where}: study archive has no valid size or SHA-256")
    require(archive.get("manifest_sha256") == entry["study_manifest"]["sha256"],
            f"{where}: copied study manifest differs from the archive descriptor")
    require(inventory.get("schema") == STUDY_ARTIFACT_SCHEMA
            and inventory.get("plan_sha256") == canonical(plan)
            and all(inventory.get(key) == plan[key] for key in ("dataset", "method", "source_commit", "adapter_commit")),
            f"{where}: study manifest belongs to another plan or source revision")
    entries = inventory.get("files")
    require(isinstance(entries, list) and entries, f"{where}: original study file inventory is empty")
    files = {}
    for item in entries:
        require(isinstance(item, dict), f"{where}: invalid original study file descriptor")
        name = item.get("path")
        require(isinstance(name, str) and name and "\\" not in name, f"{where}: invalid original study file name")
        path = PurePosixPath(name)
        require(path.parts and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
                and name not in files, f"{where}: unsafe or duplicate original study file name: {name}")
        require(type(item.get("bytes")) is int and item["bytes"] >= 0
                and isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                f"{where}: original study file has no valid size or SHA-256: {name}")
        files[name] = item
    hpo = {f"hpo/trial-{i:03d}" for i in range(len(plan["trials"]))}
    expected = {(seed, split) for seed in plan["seed_ids"] for split in plan.get("official_splits", [0])}
    finals = {f"final/seed-{seed}/split-{split}" for seed, split in expected}
    require(inventory.get("hpo_attempts") == 48 and type(inventory.get("selected_trial")) is int
            and 0 <= inventory["selected_trial"] < 48, f"{where}: incomplete HPO or invalid selected trial")
    for area, prefixes in (("hpo", hpo), ("final", finals)):
        actual = {name.removesuffix("/attempt.json") for name in files
                  if name.startswith(area + "/") and name.endswith("/attempt.json")}
        require(actual == prefixes, f"{where}: original {area} attempts do not cover the registered study")
    for prefix in hpo | finals:
        for name in ("attempt.json", "plan.json", "configuration.json", "invocation.json",
                     "ARTIFACT_MANIFEST.json", "upstream_source.tar", "adapter_source.tar"):
            require(files.get(f"{prefix}/{name}", {}).get("bytes", 0) > 0,
                    f"{where}: original attempt is missing {prefix}/{name}")
    for prefix in finals:
        for name in ("development_selection.json", "adapter_output/receipt.json"):
            require(files.get(f"{prefix}/{name}", {}).get("bytes", 0) > 0,
                    f"{where}: original final fit is missing {prefix}/{name}")
    fits = inventory.get("final_fits")
    require(isinstance(fits, list), f"{where}: final-fit inventory is missing")
    seen = set()
    for fit in fits:
        require(isinstance(fit, dict), f"{where}: invalid final fit")
        pair = (fit.get("seed"), fit.get("split"))
        require(pair in expected and pair not in seen, f"{where}: unexpected or duplicate final seed/split")
        seen.add(pair)
        prefix = f"final/seed-{pair[0]}/split-{pair[1]}/adapter_output/"
        artifacts = fit.get("artifacts") or {}
        distinct = set()
        for kind in ("predictions", "checkpoint", "split_manifest", "timing_trace"):
            item = artifacts.get(kind)
            require(isinstance(item, dict), f"{where}: final fit omits {kind}")
            name = item.get("path")
            require(isinstance(name, str) and name.startswith(prefix) and name in files and name not in distinct
                    and item.get("bytes", 0) > 0
                    and all(item.get(key) == files[name].get(key) for key in ("bytes", "sha256")),
                    f"{where}: final {kind} does not identify its original study bytes")
            distinct.add(name)
    require(seen == expected, f"{where}: final fits omit registered seeds or official splits")


def check_run_manifest(folder, stem, dataset, artifact):
    """Check copied memberships, all source identities and Roman's ten splits."""
    value = read_json(inside_file(folder, f"{stem}.json"))
    require(value.get("schema") == "cross-morse-run-manifest-1" and value.get("dataset") == dataset,
            f"{stem}: executed manifest has the wrong schema or dataset")
    require(value.get("scientific_run") is True and value.get("is_benchmark_data") is True
            and value.get("profile") == "full", f"{stem}: executed manifest is not a full benchmark run")
    identity = artifact.get("identity") or {}
    if dataset == "Roman-empire":
        campaign = value.get("campaign") or {}
        parts = campaign.get("per_split") or []
        require(identity.get("campaign") is True and len(parts) == 10
                and sorted(p.get("split_index", -1) for p in parts) == list(range(10)),
                "roman_empire: all ten distinct official splits are required")
        expected_parts = identity.get("splits") or {}
        require(len(expected_parts) == 10, "roman_empire: published campaign lacks ten identities")
        for part in parts:
            name = part.get("run_name")
            require(name in expected_parts, f"roman_empire: unregistered split directory {name}")
            _check_run_part(folder, part, expected_parts[name], f"{stem}_{name}", campaign=True)
    else:
        require(not value.get("campaign") and not identity.get("campaign"), f"{stem}: unexpected split campaign")
        _check_run_part(folder, value, identity, stem, campaign=False)
    seeds = (value.get("seeds") or {}).get("training_seeds")
    require(isinstance(seeds, list) and len(seeds) == 12 and len(set(seeds)) == 12
            and all(type(s) is int and s >= 0 for s in seeds), f"{stem}: twelve executed optimization seeds are required")
    return value


def _check_run_part(folder, part, expected, stem, *, campaign):
    identities = part.get("identities") or {}
    for key in ("source_sha256", "config_sha256", "unit_sha256"):
        require(isinstance(identities.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", identities[key])
                and identities[key] == expected.get(key), f"{stem}: {key} differs from the published run")
    arrays = part.get("arrays") if campaign else (part.get("splits") or {}).get("arrays")
    require(isinstance(arrays, dict) and "splits.npz" in arrays, f"{stem}: executed role memberships are missing")
    require(arrays["splits.npz"].get("sha256") == identities["unit_sha256"],
            f"{stem}: membership identity differs from the run")
    copied = part.get("index_arrays_copied") or []
    require(set(copied) == {f"{stem}_{name}" for name in arrays}, f"{stem}: incomplete copied membership arrays")
    for name, descriptor in arrays.items():
        path = inside_file(folder, f"{stem}_{name}")
        check_digest(path, descriptor.get("sha256"))
        require(path.stat().st_size == descriptor.get("bytes"), f"{stem}: membership byte count differs")


def check_observed_environment(folder, stem, dataset, artifact):
    value = read_json(inside_file(folder, f"{stem}.json"))
    require(value.get("schema") == "cross-morse-observed-environment-1" and value.get("dataset_run") == stem,
            f"{stem}: observed environment has the wrong schema or dataset")
    members = value.get("members")
    require(isinstance(members, list) and members, f"{stem}: observed environments are empty")
    expected = ({f"{stem}/{name}" for name in artifact.get("identity", {}).get("splits", {})}
                if dataset == "Roman-empire" else {stem})
    require(len(members) == len(expected) and {m.get("run") for m in members} == expected,
            f"{stem}: observed environments do not cover the published runs")
    for member in members:
        environment = member.get("environment")
        require(isinstance(environment, dict) and all(environment.get(k) for k in ("python", "torch", "device", "platform")),
                f"{member.get('run')}: observed Python, PyTorch, device and platform are required")
    return value


def check_source_data(root, manifest, display_ids):
    """Recompute completion from the bound report, rejecting empty or stale views."""
    from scripts.reconstruct_paper import summarize
    from scripts.reconstruct import source_inventory
    folder = Path(root) / "paper/source_data"
    index = check_snapshot(root, folder, manifest, check_builders=True)
    require(index.get("schema") == "cross-morse-source-data-1", "source-data index has the wrong schema")
    reports = index.get("reports") or {}
    loaded = {}
    for key in ("reconstruction", "workspace_files", "status"):
        descriptor = reports.get(key)
        require(isinstance(descriptor, dict), f"source data has no {key} report")
        name = descriptor.get("path")
        path = inside_file(folder, name)
        check_digest(path, descriptor.get("sha256"))
        require(index["files"].get(name) == descriptor["sha256"], f"source-data inventory omits {key} report")
        loaded[key] = read_json(path)
    result = loaded["reconstruction"]
    require(result.get("schema") == "cross-morse-reconstruction-1", "reconstruction report has the wrong schema")
    actual_ids = [d.get("id") for d in result.get("displays", [])]
    require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(display_ids),
            "published reconstruction does not account for every registered display exactly once")
    require(not result.get("synthetic_examples"), "published source data is a synthetic example")
    inventory = loaded["workspace_files"].get("files") or {}
    reconstruction_name = reports["reconstruction"]["path"]
    require((inventory.get("reconstruction.json") or {}).get("sha256") == index["files"][reconstruction_name],
            "reconstruction report differs from its workspace inventory")
    source_files = {f"displays/{name}": digest for name, digest in index["files"].items()
                    if name.startswith(("tables/", "data/", "compare/"))}
    required_sources = {name: item["sha256"] for name, item in inventory.items()
                        if name.startswith(("displays/tables/", "displays/data/", "displays/compare/"))
                        and Path(name).suffix in (".csv", ".json")}
    require(source_files and source_files == required_sources,
            "published numerical source files do not match the reconstruction inventory")
    _check_display_sources(folder, index, result, inventory)
    expected = summarize(result, manifest)
    require(loaded["status"] == expected == index.get("reconstruction"),
            "published completion summary differs from its reconstruction report")
    require(index.get("software_sources") == result.get("software_sources") == source_inventory(Path(root)),
            "source-data software identities differ from the reconstruction report or current source inventory")
    inputs = result.get("inputs")
    require(isinstance(inputs, list), "reconstruction report has no input inventory")
    require(index.get("inputs_sha256") == hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            "source-data input identity differs from the reconstruction report")
    return index, expected


def _check_display_sources(folder, index, result, inventory):
    """Bind display completion and comparison counts to the published numerical files.

    The committed view omits rendered images, but their identities must still be present in the original
    workspace inventory. Numerical source and comparison CSV files are copied into the view and read here.
    Partial, honestly described reconstructions remain valid views; fulfillment is a separate decision.
    """
    from paper.build.displays.framework import COMPARISON_STATES
    outputs = result.get("outputs")
    require(isinstance(outputs, list), "reconstruction report has no generated output inventory")
    by_id = {}
    for output in outputs:
        require(isinstance(output, dict) and isinstance(output.get("id"), str)
                and output["id"] not in by_id, "generated outputs have invalid or duplicate identities")
        by_id[output["id"]] = output
    total = {state: 0 for state in COMPARISON_STATES}
    comparisons_seen = set()
    for display in result["displays"]:
        ident = display["id"]
        output_id = f"display:{ident}"
        output = by_id.get(output_id)
        full = display.get("full_numerical_display") is True
        if output is None:
            require(not full and display.get("content_complete") is not True
                    and display.get("comparison_complete") is not True,
                    f"{ident}: completion flags have no generated display output")
            require(display.get("comparison") in (None, dict.fromkeys(COMPARISON_STATES, 0)),
                    f"{ident}: comparisons have no generated display output")
            continue
        require(output.get("display_ids") == [ident] and output_id in display.get("generated_outputs", []),
                f"{ident}: generated output is not linked to this display")
        require(full == (output.get("coverage") == "full_numerical_display"),
                f"{ident}: display completion differs from its generated output")
        for key in ("content_complete", "comparison_complete", "comparison_coverage"):
            require(display.get(key) == output.get(key), f"{ident}: {key} differs from its generated output")
        files, hashes = output.get("files"), output.get("file_sha256")
        require(isinstance(files, list) and files and len(files) == len(set(files))
                and isinstance(hashes, dict) and set(hashes) == set(files),
                f"{ident}: generated output has no exact file inventory")
        copied, rendered = [], []
        for name in files:
            require(isinstance(name, str) and name.startswith("displays/")
                    and "\\" not in name and ".." not in PurePosixPath(name).parts
                    and PurePosixPath(name).as_posix() == name
                    and isinstance(inventory.get(name), dict), f"{ident}: output is absent from workspace FILES: {name}")
            descriptor = inventory[name]
            require(hashes[name] == descriptor.get("sha256") and isinstance(hashes[name], str)
                    and re.fullmatch(r"[0-9a-f]{64}", hashes[name])
                    and type(descriptor.get("bytes")) is int and descriptor["bytes"] > 0,
                    f"{ident}: generated output identity differs from workspace FILES: {name}")
            relative = name.removeprefix("displays/")
            if relative.startswith(("tables/", "data/", "compare/")) and Path(name).suffix in (".csv", ".json"):
                require(index["files"].get(relative) == hashes[name],
                        f"{ident}: generated numerical output is absent or changed in the published source data: {name}")
                path = inside_file(folder, relative)
                require(path.stat().st_size == descriptor["bytes"], f"{ident}: source-data byte count differs: {name}")
                copied.append(relative)
            if relative.startswith("figures/") and Path(name).suffix in (".svg", ".png", ".pdf"):
                rendered.append(name)
        counts = {state: 0 for state in COMPARISON_STATES}
        comparison_files = [name for name in copied if name.startswith("compare/") and name.endswith(".csv")]
        for name in comparison_files:
            require(name not in comparisons_seen, f"{ident}: comparison file is assigned to more than one display")
            comparisons_seen.add(name)
            with inside_file(folder, name).open(encoding="utf-8", newline="") as handle:
                rows = csv.DictReader(handle)
                require(rows.fieldnames == ["group", "path", "computed", "reported", "status", "note"],
                        f"{ident}: comparison CSV has the wrong columns")
                for row in rows:
                    require(row.get("status") in counts and None not in row,
                            f"{ident}: comparison CSV has an invalid status or row")
                    counts[row["status"]] += 1
        require(display.get("comparison") == output.get("comparison") == counts,
                f"{ident}: comparison counts differ from the published comparison CSV")
        for key, value in counts.items():
            total[key] += value
        if full:
            require(output.get("content_complete") is True and output.get("comparison_complete") is True
                    and output.get("input_kind") != "synthetic_examples",
                    f"{ident}: full display lacks complete content and comparisons")
            require(any(name.startswith(("data/", "tables/")) for name in copied),
                    f"{ident}: full display has no published source data")
            for name in copied:
                if name.startswith(("data/", "tables/")) and name.endswith(".csv"):
                    with inside_file(folder, name).open(encoding="utf-8", newline="") as handle:
                        reader = csv.reader(handle)
                        columns = next(reader, [])
                        first = next(reader, None)
                        require(columns and first is not None and len(first) == len(columns)
                                and all(len(row) == len(columns) for row in reader),
                                f"{ident}: full display has an empty or malformed source CSV: {name}")
            if ":Table " in ident:
                require(any(name.startswith("tables/") and name.endswith(".csv") for name in copied),
                        f"{ident}: full table has no generated table CSV")
            if ":Figure " in ident:
                require(rendered, f"{ident}: full figure has no rendered file in workspace FILES")
            illustration = ident == "main:Figure 1" and output.get("scope") == "illustration"
            require(illustration or comparison_files and sum(counts.values()) > 0,
                    f"{ident}: full numerical/protocol display has no actual comparisons")
            require(not any(counts[state] for state in ("not_comparable", "no_reference", "missing")),
                    f"{ident}: full display has missing or incomparable required values")
    expected_comparisons = {name for name in index["files"] if name.startswith("compare/") and name.endswith(".csv")}
    require(comparisons_seen == expected_comparisons,
            "published comparison files are not accounted for by the generated displays")
    generators = result.get("display_generators") or {}
    require(generators.get("comparison_with_printed_values") == total,
            "global comparison counts differ from the published comparison CSV files")
    require(generators.get("comparison_coverage_complete") is
            all(display.get("comparison_complete") is True for display in result["displays"]),
            "global comparison completeness differs from the individual displays")


def reconstruction_problems(summary, display_count):
    """Keep failure and absent comparison metadata distinct from exact agreement."""
    problems = []
    if summary.get("displays") != display_count or summary.get("complete") != display_count or summary.get("incomplete"):
        problems.append("some registered displays are incomplete")
    if type(summary.get("values_exact")) is not int or summary.get("values_exact", 0) <= 0:
        problems.append("no measured values have been compared with the paper")
    for key, label in (("values_differ", "values differ from the paper"),
                       ("values_not_comparable", "values are not comparable"),
                       ("values_no_reference", "required values have no paper reference"),
                       ("values_missing", "required printed values were not computed")):
        if type(summary.get(key)) is not int or summary[key] != 0:
            problems.append(label)
    if summary.get("comparison_coverage_complete") is not True:
        problems.append("the comparison does not cover every required printed value")
    if summary.get("display_content_incomplete"):
        problems.append("required display cells or panels are incomplete")
    for key in ("generator_errors", "module_errors", "artifacts_awaiting_publication"):
        if summary.get(key):
            problems.append(key.replace("_", " "))
    return problems
