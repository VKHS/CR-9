"""Dexplore-only label allocation and complete open-loop proposal coverage."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from cross_morse.runtime import is_graph_collection
from .artifacts import canonical
from .protocol import OPEN_LOOP


def development_partition(data, *, seed, max_graphs=0, selection_fraction=.25):
    if not 0 < selection_fraction < 1 or max_graphs < 0:
        raise ValueError('Invalid development split settings')
    rng = np.random.default_rng(seed)
    if is_graph_collection(data):
        ids = sorted(map(int, data.graph_splits['explore']))
        if max_graphs:
            ids = ids[:max_graphs]
        if len(ids) < 3:
            raise ValueError('Need at least two fitting graphs and one configuration-selection graph')
        order = rng.permutation(ids)
        count = min(len(ids)-2, max(1, int(round(len(ids)*selection_fraction))))
        select = set(map(int, order[:count]))
        items = [{'episode_id': i, 'fit_ids': [] if i in select else data.graph(i).splits['explore'].tolist(),
                  'selection_ids': data.graph(i).splits['explore'].tolist() if i in select else []}
                 for i in ids]
        grouping = 'complete input graph; no graph shared by fitting and configuration selection'
    else:
        ids = np.asarray(data.splits['explore'], dtype=np.int64)
        if len(ids) < 2:
            raise ValueError('Need at least two development labels')
        order = rng.permutation(ids)
        count = min(len(ids)-1, max(1, int(round(len(ids)*selection_fraction))))
        items = [{'episode_id': 0, 'fit_ids': sorted(map(int, order[count:])),
                  'selection_ids': sorted(map(int, order[:count]))}]
        grouping = ('one transductive input; disjoint scored development labels, not independent input graphs; '
                    'checkpoint diversity cannot establish unseen-input generalization')
    for item in items:
        if set(item['fit_ids']) & set(item['selection_ids']):
            raise ValueError('Development-role overlap')
    return {'schema': 'cm-arm-development-split-1', 'source_role': 'Dexplore', 'seed': int(seed),
            'selection_fraction': selection_fraction, 'episodes': items, 'grouping': grouping,
            'unique_input_contexts': len(items), 'independent_input_collection': is_graph_collection(data),
            'ids_are_local_prediction_rows': True}


def subset_labels(parent, episode, role, ids):
    allowed = np.asarray(episode.splits[role], dtype=np.int64)
    lookup = {int(n): i for i, n in enumerate(allowed)}
    if len(set(map(int, ids))) != len(ids) or not set(map(int, ids)) <= set(lookup):
        raise ValueError('Requested labels lie outside the permitted role')
    with parent._label_scope(role):
        labels = episode.labels(role)
    return labels[np.asarray([lookup[int(n)] for n in ids], dtype=np.int64)]


def open_loop_family(root, comparator, *, depth, proposal_limit=0):
    """Every registered open-loop behaviour at the paired ensemble, plus qC.

    No action-ID sorting/truncation. A nonzero limit is a safety ceiling that
    raises, never permission to discard the strongest comparator or a family.
    """
    root = Path(root)
    value = json.loads((root/'fit/registry.json').read_text())
    registry = value if isinstance(value, list) else value.get('candidates', value.get('registry', []))
    if not isinstance(registry, list) or not registry:
        raise ValueError('Missing frozen candidate registry')
    chosen = {}
    for item in [*registry, comparator]:
        if item.get('route') not in OPEN_LOOP or item.get('ensemble') != comparator['ensemble']:
            continue
        spec = copy.deepcopy(item)
        if spec.get('sequence') is not None and 9 in spec['sequence']:
            raise ValueError('Open-loop registry contains dynamic reset A9')
        # Hash the actual candidate, not just the resulting action sequence:
        # a learned stop/MPC-OL has a distinct inference cost and policy law.
        key = canonical(spec)
        chosen[key] = spec
    if comparator.get('route') not in OPEN_LOOP or not chosen:
        raise ValueError('The locked comparator must belong to the open-loop family')
    present = {r['route'] for r in chosen.values()}
    expected = {r.get('route') for r in registry if r.get('route') in OPEN_LOOP
                and r.get('ensemble') == comparator['ensemble']}
    if not expected <= present:
        raise AssertionError('Open-loop family coverage lost')
    if proposal_limit and proposal_limit < len(chosen):
        raise ValueError(f'Oracle proposal ceiling {proposal_limit} would discard required family members ({len(chosen)}); use 0 for all')
    rows = [dict(configuration_id=k[:24], candidate=v, source='locked_comparator' if v == comparator else 'registered_open_loop')
            for k, v in sorted(chosen.items())]
    return rows, {'families': sorted(present), 'candidate_count': len(rows),
                  'locked_comparator_included': any(v['candidate'] == comparator for v in rows),
                  'arbitrary_truncation': False, 'controlled_depth': depth,
                  'variable_terminal_depths_preserved': True}
