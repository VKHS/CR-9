"""Predeclared budget-matched experiment contracts.

An expected deployment-work ceiling is not Algorithm S3's per-input hard cap.
Both use the same instrumented product unit here; neither is a latency promise.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from .artifacts import canonical

PAPER_CEILINGS = {
    'ogbn-arxiv': 1.15, 'ogbn-proteins': 1.17, 'Roman-empire': 1.30,
    'PascalVOC-SP': 1.29, 'COCO-SP': 1.33, 'Peptides-func': 1.18,
}
OPEN_LOOP = ('nominal', 'best_depth', 'learned_stop', 'phase', 'fixed_sequence', 'mpc_ol')
ARMS = ('E3', 'E4', 'E5', 'E6')


def finite(value, name, low=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise ValueError(f'{name} must be finite and >= {low}')
    return float(value)


@dataclass(frozen=True)
class BudgetContract:
    dataset: str
    expected_ratio: float
    hard_cap_ratio: float
    tolerance: float = .01
    cost_basis: str = 'deployed_multiply_add_products'
    cap_origin: str = 'coupled_to_expected_ceiling'

    def __post_init__(self):
        finite(self.expected_ratio, 'expected ratio', 1.)
        finite(self.hard_cap_ratio, 'S3 hard-cap ratio', 1.)
        finite(self.tolerance, 'budget tolerance')
        if self.tolerance > .05 or self.cost_basis != 'deployed_multiply_add_products':
            raise ValueError('Invalid tolerance or work coordinate')

    def record(self):
        return {**asdict(self),
                'algorithm': 'Supplementary Algorithm S3: full comparator restart reserve through final prediction',
                'expected_constraint': 'ratio of equal-input mean deployed work; development-selected, independent-comparison-audited',
                'hard_constraint': 'per-input, per-ensemble S3 operation bounds; independent of the expected ceiling',
                'qualification': 'excluded only when separately counted; all inference sensing, decoding, search and restart work charged',
                'work_limit': 'instrumented sparse/dense multiply-add products, not complete FLOPs, time or energy'}


def budget_contract(dataset, *, expected=None, hard_cap=None, tolerance=.01, fixture=False):
    if expected is None:
        if dataset not in PAPER_CEILINGS:
            if not fixture:
                raise ValueError(f'No S46 ceiling for {dataset}; explicitly register a separate experiment')
            expected = 1.33  # Test-only input; never inferred as a manuscript ceiling.
        else:
            expected = PAPER_CEILINGS[dataset]
    return BudgetContract(dataset, finite(expected, 'expected ratio', 1.),
                          finite(expected if hard_cap is None else hard_cap, 'hard cap', 1.),
                          finite(tolerance, 'tolerance'), cap_origin=(
                              'coupled_to_expected_ceiling' if hard_cap is None else 'explicit_predeclared_separate_cap'))


def terminal_reserve_requirement(controller_work, prediction_work, restart_bound):
    """Necessary S3 completion ceiling: C_success - C_prediction + W_C.

    This is not sufficient: upper-bound prechecks, support and the control path
    can require more. It does not count an unused reserve as executed work.
    """
    total = finite(controller_work, 'successful controller work')
    final = finite(prediction_work, 'final prediction work')
    reserve = finite(restart_bound, 'restart bound')
    if final > total or final > reserve:
        raise ValueError('Prediction must be a subset of success work and fit in the reserve')
    return {'necessary_cap': total - final + reserve,
            'success_work': total, 'final_prediction_work': final, 'restart_bound': reserve,
            'is_sufficient': False, 'unused_reserve_is_not_executed_work': True}


def deployed_work(trace):
    """Use the runtime's exact qualification subtraction, never a guessed factor."""
    total = finite(trace['multiply_add_work'], 'total work')
    qualification = finite(trace.get('qualification_multiply_add_work', 0.), 'qualification work')
    deployed = finite(trace.get('deployed_multiply_add_work', total - qualification), 'deployment work')
    if abs(total - qualification - deployed) > 1e-8 * max(1., total):
        raise ValueError('Deployment and qualification work do not reconcile')
    return deployed


def candidate_id(spec):
    return canonical(spec)[:24]


def choose_development(rows, contract):
    """Deterministic choice using Dexplore selection observations only.

    A candidate that failed, lacked support, or exceeded the registered mean
    budget cannot win. No confirmation loss enters this API.
    """
    if not rows:
        raise ValueError('No development candidates')
    ids = [r['configuration_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate development configuration')
    eligible = []
    for row in rows:
        if row.get('role') != 'Dexplore_selection':
            raise ValueError('Configuration selection accepts only Dexplore_selection')
        if row.get('status') != 'EXECUTED' or not row.get('support_qualified', False):
            continue
        loss = finite(row['loss'], 'development loss')
        work = finite(row['mean_work'], 'development work')
        reference = finite(row['mean_qc_work'], 'reference work', 1e-12)
        if work <= reference * contract.expected_ratio * (1 + contract.tolerance) + 1e-8:
            eligible.append(row)
    if not eligible:
        return {'selected': None, 'status': 'NO_FEASIBLE_CONFIGURATION', 'eligible': [],
                'rule': 'minimum development loss subject to support and expected-work ceiling; tie: configuration ID'}
    winner = min(eligible, key=lambda r: (r['loss'], r['configuration_id']))
    return {'selected': winner['configuration_id'], 'status': 'SELECTED',
            'eligible': sorted(r['configuration_id'] for r in eligible),
            'rule': 'minimum development loss subject to support and expected-work ceiling; tie: configuration ID'}
