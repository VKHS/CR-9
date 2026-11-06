"""Reconcile recorded member work before making a common-ceiling comparison."""
from __future__ import annotations
import math

ARMS = ('qC', 'feedback', 'E3', 'E4', 'E5', 'E6')


def summarize_work(timings, traces, *, ratio, scored, basis='qualification_inclusive'):
    if basis not in ('deployment', 'qualification_inclusive'):
        raise ValueError('Unknown work coordinate')
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('A finite positive common-ceiling ratio is required')
    samples = {}
    for row in timings:
        key = (int(row['episode_id']), row['arm'])
        if row['arm'] not in ARMS:
            raise ValueError('Unknown work-ledger arm')
        value = float(row['multiply_add_work'])
        wall = float(row['wall_seconds'])
        if not math.isfinite(value) or value < 0 or not math.isfinite(wall) or wall < 0:
            raise ValueError('Work and elapsed time must be finite and nonnegative')
        item = samples.setdefault(key, {'executed_work': 0., 'deployed_work': 0., 'embedded_qualification_work': 0., 'qualification_work': 0., 'wall_seconds': 0.})
        item['executed_work'] += value
        deployment = float(row.get('deployed_multiply_add_work', value))
        if not math.isfinite(deployment) or not 0 <= deployment <= value:
            raise ValueError('Invalid deployment-work subtraction')
        item['deployed_work'] += deployment
        item['embedded_qualification_work'] += value - deployment
        item['wall_seconds'] += wall
    for row in traces:
        qa = row.get('actual_state_support_audit')
        if qa is None:
            continue
        key = (int(row['episode_id']), row['arm'])
        if key not in samples:
            raise ValueError('Qualification trace lacks corresponding timed execution')
        value = float(qa['qualification_multiply_add_work'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid support-audit work')
        samples[key]['qualification_work'] += value
    episodes = sorted({i for i, _ in samples})
    rows = []
    for eid in episodes:
        if (eid, 'qC') not in samples or samples[eid, 'qC']['executed_work'] <= 0:
            raise ValueError('Each input needs a positive observed comparator-work reference')
        reference = samples[eid, 'qC']['deployed_work' if basis == 'deployment' else 'executed_work']
        if reference <= 0:
            raise ValueError('Comparator deployment work must be positive')
        for arm in ARMS:
            if (eid, arm) not in samples:
                raise ValueError('Missing arm work record for an input')
            value = samples[eid, arm]
            total = value['executed_work'] + value['qualification_work']
            # Keep both coordinates visible. Development-selected studies compare
            # deployment against deployment; the older inclusive coordinate is
            # retained only for explicitly legacy records.
            rows.append({'episode_id': eid, 'arm': arm, **value,
                         'qualification_inclusive_work': total,
                         'comparison_work': value['deployed_work'] if basis == 'deployment' else total,
                         'comparator_reference_work': reference, 'ceiling': ratio * reference,
                         'within_inclusive_ceiling': total <= ratio * reference + 1e-8,
                         'within_declared_ceiling': (value['deployed_work'] if basis == 'deployment' else total) <= ratio * reference + 1e-8})
    summaries = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        usable = bool(scored.get(arm, False))
        inclusive = sum(r['qualification_inclusive_work'] for r in selected)
        total = sum(r['comparison_work'] for r in selected)
        reference = sum(r['comparator_reference_work'] for r in selected)
        summaries[arm] = {
            'scored_all_inputs': usable,
            'mean_qualification_inclusive_work': inclusive / len(selected),
            'mean_comparison_work': total / len(selected),
            'mean_deployed_work': sum(r['deployed_work'] for r in selected) / len(selected),
            'mean_comparator_reference_work': reference / len(selected),
            'observed_work_ratio': total / reference,
            'within_mean_ceiling': usable and total <= ratio * reference + 1e-8,
            'within_every_input_ceiling': usable and all(r['within_declared_ceiling'] for r in selected),
            'over_ceiling_inputs': sum(not r['within_declared_ceiling'] for r in selected),
        }
    return {'work_unit': 'instrumented multiply-add products', 'budget_ratio': ratio,
            'cost_basis': basis,
            'scope': ('deployment products with measured qualification recorded separately; not complete FLOPs or wall time'
                      if basis == 'deployment' else 'legacy qualification-inclusive comparison; not manuscript deployment cost'),
            'weighting': 'equal input episodes, fixed ensemble; one transductive graph is one input',
            'inference_work_and_qualification_are_separate': True,
            'per_input': rows, 'summary': summaries}
