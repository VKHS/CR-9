"""Shared, label-free execution for development selection and checkpoint replay."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from cross_morse.learning import Controller
from cross_morse.training import load_plant
from cross_morse.runtime import graph_operator
from cross_morse.policies import execute_policy
from .execution import execute_arm, execute_budget_ensemble, audit_sequence
from .protocol import deployed_work


class StudyBackend:
    def __init__(self, root, data, config, teacher, comparator, member_seeds, device, decoders, *, costs=None):
        self.root, self.data, self.config = Path(root), data, config
        self.teacher, self.comparator = teacher, comparator
        self.members, self.device, self.decoders = list(member_seeds), device, decoders
        self.models = {s: load_plant(data, config, self.root/'train', s, device) for s in self.members}
        self.reference = dict(np.load(self.root/'explore/nominal_reference.npz', allow_pickle=False))
        self.costs = costs

    def controller(self, candidate):
        profile = candidate.get('profile', 'full')
        return Controller.load(self.root/f"fit/{'full' if profile == 'none' else profile}/controller.pt", 'cpu')

    def execute(self, episode, variant, *, qc_reference_work=None, hard_cap_ratio=None):
        """Return predictions, actual traces and costs, without accessing labels."""
        graph = graph_operator(episode, self.device)
        x = episode.x.to(self.device)
        kind = variant['kind']; candidate = variant.get('candidate', self.teacher)
        values, traces = [], []
        if kind == 's3':
            if qc_reference_work is None or hard_cap_ratio is None:
                raise ValueError('S3 requires a registered cap coordinate and paired comparator work')
            p, trace = execute_budget_ensemble(
                [self.models[s] for s in self.members], x, graph, self.config,
                [self.controller(candidate) for _ in self.members],
                [self.controller({'profile': 'lean'}) for _ in self.members],
                candidate, self.comparator, [self.controller(self.comparator) for _ in self.members],
                self.reference, budget=hard_cap_ratio * qc_reference_work,
                per_member_costs=None if self.costs is None else [self.costs for _ in self.members])
            trace['member_seed'] = None
            traces.append(trace); values.append(p)
        else:
            for seed in self.members:
                model = self.models[seed]
                control = self.controller(candidate)
                if kind == 'policy':
                    p, trace = execute_policy(model, x, graph, self.config, candidate, control, self.reference)
                elif kind in ('decoder', 'predicted_state'):
                    arm = variant['arm']
                    decoder = self.decoders[variant['decoder_id']] if kind == 'decoder' else None
                    p, trace = execute_arm(arm, model, x, graph, self.config, control, candidate, decoder=decoder)
                    audited, qa = audit_sequence(model, x, graph, self.config, control,
                                                 trace['actions'], trace['safe_depths'])
                    if not np.array_equal(audited, p):
                        raise ValueError('Actual-state support audit changed the frozen predictions')
                    trace['actual_state_support_audit'] = qa
                else:
                    raise ValueError('Unknown arm variant')
                trace['member_seed'] = seed
                traces.append(trace); values.append(p)
        qualified = all(t.get('feasible', True) and t.get('actual_state_support_audit', {}).get('feasible', True)
                        for t in traces)
        total = sum(float(t['multiply_add_work']) for t in traces)
        inference = sum(deployed_work(t) for t in traces)
        extra = sum(float(t.get('actual_state_support_audit', {}).get('qualification_multiply_add_work', 0))
                    for t in traces)
        result = {'traces': traces, 'executed_work': total, 'deployed_work': inference,
                  'embedded_qualification_work': total-inference, 'separate_qualification_work': extra,
                  'wall_seconds': sum(float(t['wall_seconds']) for t in traces),
                  'support_qualified': qualified, 'kind': kind,
                  'status': 'INFEASIBLE' if any(p is None for p in values) else 'EXECUTED'}
        if any(p is None for p in values):
            result['prediction'] = None
        else:
            result['prediction'] = np.mean(np.stack(values).astype(np.float64), axis=0).astype(np.float32)
        return result


def variants(config, teacher, family, decoder_ids):
    """Predeclare a nontruncated search family with explicit comparator fallback.

    E3 includes every original open-loop candidate, not only its learned decoder.
    Selecting one of those alternatives is recorded as a direct open-loop result.
    """
    from .protocol import candidate_id
    result = {arm: [] for arm in ('E3', 'E4', 'E5', 'E6')}
    for arm in ('E3', 'E4'):
        for decoder_id in decoder_ids[arm]:
            result[arm].append({'kind': 'decoder', 'arm': arm, 'decoder_id': decoder_id,
                                'candidate': copy.deepcopy(teacher)})
    for entry in family:
        result['E3'].append({'kind': 'policy', 'arm': 'E3', 'candidate': entry['candidate'],
                             'origin': entry['source']})
    qc = next((r['candidate'] for r in family if r['source'] == 'locked_comparator'), None)
    if qc is None:
        raise ValueError('Locked comparator missing from proposal family')
    for arm in ('E4', 'E5', 'E6'):
        result[arm].append({'kind': 'policy', 'arm': arm, 'candidate': copy.deepcopy(qc),
                           'origin': 'locked_comparator', 'policy_interpretation': 'comparator retained; no arm improvement claimed'})
    horizons = sorted({int(teacher.get('horizon', 1)), *map(int, config['controller']['horizons'])})
    for h in horizons:
        candidate = dict(teacher, horizon=min(h, config['model']['depth']))
        result['E5'].append({'kind': 'predicted_state', 'arm': 'E5', 'candidate': candidate})
    result['E6'].append({'kind': 's3', 'arm': 'E6', 'candidate': copy.deepcopy(teacher),
                         'certificate_mode': 'acquisition unless separately proven; no fabricated radius/invariance premise'})
    for arm, rows in result.items():
        seen = set(); out = []
        for row in rows:
            key = candidate_id(row)
            if key not in seen:
                out.append(dict(row, configuration_id=key)); seen.add(key)
        result[arm] = sorted(out, key=lambda r: r['configuration_id'])
    return result
