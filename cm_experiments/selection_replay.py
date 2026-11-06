"""Numerically reconstruct development choices before accepting independent comparison."""
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
from cross_morse.tasks import task_spec, sample_losses
from .protocol import BudgetContract, choose_development, deployed_work
from .artifacts import sha256, canonical


def replay_selection(root, experiment, config):
    root = Path(root)
    selection = json.loads((root/'development_selection.json').read_text())
    partition = json.loads((root/'development_partition.json').read_text())
    registry = json.loads((root/'variant_registry.json').read_text())
    lock = json.loads((root/'selection_lock.json').read_text())
    if lock['development_selection_sha256'] != sha256(root/'development_selection.json') or \
       lock['variant_registry_sha256'] != sha256(root/'variant_registry.json'):
        raise ValueError('The comparison execution has a different development lock')
    if selection['registry_sha256'] != canonical(registry) or selection['partition_sha256'] != canonical(partition):
        raise ValueError('Development inputs changed')
    if not lock.get('locked_before_comparison_label_access') or selection.get('comparison_used'):
        raise ValueError('Configuration selection is not locked before D_compare')
    keys = {f.name for f in fields(BudgetContract)}
    contract = BudgetContract(**{k: v for k, v in experiment['budget'].items() if k in keys})
    checked = 0
    for arm, rows in selection['candidates'].items():
        ids = {v['configuration_id'] for v in registry[arm]}
        if {r['configuration_id'] for r in rows} != ids:
            raise ValueError('Development rows do not cover the complete frozen search family')
        for row in rows:
            observations = row['observations']
            for ob in observations:
                amount=sum(deployed_work(t) for t in ob['traces'])
                if not np.isclose(amount,ob['work'],rtol=1e-10,atol=1e-8):
                    raise ValueError('Development work is not its executed-trace sum')
            if not observations:
                raise ValueError('Missing development inputs')
            mw = float(np.mean([v['work'] for v in observations])); mq = float(np.mean([v['qc_work'] for v in observations]))
            if not np.isclose(mw, row['mean_work'], rtol=0, atol=1e-8) or not np.isclose(mq, row['mean_qc_work'], rtol=0, atol=1e-8):
                raise ValueError('Development-work means do not recompute')
            if row['status'] != 'EXECUTED':
                if row['predictions'] is not None:
                    raise ValueError('Infeasible development candidate has a completed prediction file')
                continue
            path = root/row['predictions']
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Development prediction path escapes study')
            with np.load(path, allow_pickle=False) as z:
                losses = sample_losses(z['y_pred'], z['y_true'], task_spec(config))
                input_ids = z['input_id']
                recomputed = {int(i): float(np.mean(losses[input_ids == i])) for i in np.unique(input_ids)}
            if set(recomputed) != {int(o['episode_id']) for o in observations}:
                raise ValueError('Development prediction inputs differ')
            for o in observations:
                if not np.isclose(recomputed[int(o['episode_id'])], o['loss'], rtol=1e-10, atol=1e-12):
                    raise ValueError('Development loss differs from predictions')
            if not np.isclose(np.mean(list(recomputed.values())), row['loss'], rtol=1e-10, atol=1e-12):
                raise ValueError('Development risk does not recompute')
            checked += 1
        decision = choose_development(rows, contract)
        if decision != selection['decisions'][arm] or decision != lock['decisions'][arm]:
            raise ValueError('Stored choice is not the budget-constrained development optimum')
    return {'passed': True, 'prediction_sets_recomputed': checked,
            'selection_role': 'Dexplore_selection', 'comparison_not_used': True}
