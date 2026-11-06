"""E6 paid-operation accounting with a comparator-restart reserve.

Bounds are declared before evaluation. The guarantee is about the declared
work unit, not elapsed time or GPU energy. Each callback reports observed work;
exceeding a declared bound raises an error rather than claiming a guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Any

class WorkBoundViolation(RuntimeError):
    """Observed counters invalidated a frozen operation bound."""
    def __init__(self,message,diagnostics=None):
        super().__init__(message)
        self.diagnostics=diagnostics or {}

class Unaffordable(RuntimeError):
    pass


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def action_id(value, name):
    """Do not silently turn a fractional, boolean or string action into an ID."""
    if type(value) is not int or not 0 <= value < 10:
        raise ValueError(f'{name} must be an integer action ID in [0, 9]')
    return value


@dataclass(frozen=True)
class Certificate:
    """All premises are required; a large empirical gap alone is not a proof."""
    gap: float
    lipschitz_bound: float
    radius: float
    floor: float = .002
    support_invariant: bool = False
    recovery_invariant: bool = False
    search_invariant: bool = False
    bound_source: str = ''

    def allows_skip(self):
        # +inf is permitted only for the score gap of a singleton supported set.
        # The executor verifies that set and binds this claimed gap to the
        # actual cheap-score observation before it may skip an acquisition.
        if (isinstance(self.gap, bool) or not isinstance(self.gap, (int, float))
                or math.isnan(self.gap) or self.gap < 0):
            raise ValueError('Gap must be nonnegative')
        lj = nonnegative(self.lipschitz_bound, 'Lipschitz bound')
        r = nonnegative(self.radius, 'radius'); floor = nonnegative(self.floor, 'floor')
        flags = (self.support_invariant, self.recovery_invariant, self.search_invariant)
        if not all(type(v) is bool for v in flags):
            raise ValueError('Certificate invariance fields must be booleans')
        if not isinstance(self.bound_source, str):
            raise ValueError('Certificate bound source must be a string')
        return bool(self.bound_source and all(flags) and self.gap > max(floor, 2*lj*r))


def certified_skip(score, supported, certificate):
    """Check Proposition 9 against the score actually observed at this site.

    A callback supplies regional bounds and invariance premises; it cannot
    replace the observed action gap with a larger, unrelated value. A finite
    lower bound on that gap is allowed. Infinite gaps are only certificates
    for a singleton supported set, never for invalid/missing competing scores.
    """
    if not isinstance(certificate, Certificate):
        raise ValueError('A skip certificate must be a Certificate instance')
    if not certificate.allows_skip():
        return False
    observed = score.get('gap')
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or math.isnan(observed) or observed < 0):
        raise ValueError('A skip certificate requires a nonnegative observed score gap')
    if (math.isinf(observed) or math.isinf(certificate.gap)) and len(supported) != 1:
        raise ValueError('An infinite certified gap requires singleton support')
    if certificate.gap > observed:
        raise ValueError('Certificate gap exceeds the observed score gap')
    if action_id(score.get('action'), 'Certified action') not in supported:
        raise ValueError('Certified action is outside the invariant supported set')
    return True


class WorkLedger:
    def __init__(self, budget, reserve, comparator_bound, *, work_unit='multiply_add_products'):
        self.budget = nonnegative(budget, 'budget')
        self.reserve = nonnegative(reserve, 'reserve')
        self.comparator_bound = nonnegative(comparator_bound, 'comparator bound')
        if self.reserve < self.comparator_bound:
            raise ValueError('Reserve must cover the complete comparator restart')
        if not isinstance(work_unit,str) or not work_unit:
            raise ValueError('Work unit is required')
        self.work_unit = work_unit; self.spent = 0.; self.rows = []; self.terminal = False

    def affordable(self, bound, *, terminal=False):
        bound = nonnegative(bound, 'operation bound')
        return self.spent + bound + (0 if terminal else self.reserve) <= self.budget + 1e-10

    def pay(self, operation, bound, callback: Callable[[], tuple[Any, float]], *, terminal=False, outcome='continue', depth=None):
        if self.terminal:
            raise ValueError('No operation may execute after completion')
        bound = nonnegative(bound, 'operation bound')
        if terminal and operation == 'prediction' and bound > self.reserve:
            raise ValueError('Prediction bound must fit inside the held reserve')
        if not self.affordable(bound, terminal=terminal):
            raise Unaffordable(operation)
        value, actual = callback()
        actual = nonnegative(actual, 'observed work')
        if actual > bound + 1e-9 * max(1., bound):
            raise WorkBoundViolation(f'{operation}: observed {actual} exceeds the pre-operation bound {bound}',
                {'operation':operation,'observed_work':actual,'declared_bound':bound,
                 'work_before':self.spent,'budget':self.budget,'reserve':self.reserve,
                 'status':'BOUND_VIOLATED_NO_CAP_CLAIM'})
        self.spent += actual
        self.rows.append({'operation_index': len(self.rows), 'operation': operation, 'depth': depth,
                          'paid_work': actual, 'operation_bound': bound, 'cumulative_work': self.spent,
                          'budget': self.budget, 'comparator_reserve': self.reserve,
                          'comparator_bound': self.comparator_bound, 'outcome': outcome,
                          'prechecked': True, 'work_unit': self.work_unit})
        self.terminal = terminal
        return value

    def restart(self, comparator, *, static=False):
        return self.pay('comparator_restart', self.comparator_bound, comparator, terminal=True,
                        outcome='static_fallback' if static else 'abort_to_comparator')

    def mark_infeasible(self):
        if self.spent or self.rows or self.terminal or self.budget >= self.comparator_bound:
            raise ValueError('Only an unstarted, insufficient comparator budget is infeasible')
        self.terminal = True
        self.rows.append({'operation_index': 0, 'operation': 'none', 'paid_work': 0.,
                          'cumulative_work': 0., 'budget': self.budget,
                          'comparator_reserve': self.reserve, 'comparator_bound': self.comparator_bound,
                          'outcome': 'infeasible', 'prechecked': True, 'work_unit': self.work_unit})


def run_budget_episode(backend, *, depth, budget, costs, certificate=None):
    """Execute Supplement Algorithm S3 using explicit paid backend operations.

    backend methods return (value, actual_work); no callback executes until its
    upper bound fits. score returns supported, action, recovery and gap fields.
    acquire returns the action of the full policy. predict and restart produce
    actual predictions. A certificate callback receives only the cheap score.
    """
    if type(depth) is not int or depth < 1:
        raise ValueError('Positive controlled depth required')
    names = ('initialization', 'cheap_measurement', 'score', 'acquisition', 'action', 'prediction', 'comparator')
    if set(costs) != set(names):
        raise ValueError(f'Cost schema must contain exactly {names}')
    costs = {k: nonnegative(v,k) for k,v in costs.items()}
    if costs['prediction'] > costs['comparator']:
        raise ValueError('Final prediction bound exceeds restart reserve')
    ledger = WorkLedger(budget, costs['comparator'], costs['comparator'])
    diagnostics = {'acquisitions': 0, 'certified_skips': 0, 'unaffordable_acquisitions': 0,
                   'recoveries': 0, 'certificate_assumption': 'conditional on supplied region and invariance bounds'}
    if ledger.budget < costs['comparator']:
        ledger.mark_infeasible()
        return None, {'status': 'INFEASIBLE', 'operations': ledger.rows, **diagnostics}
    if not ledger.affordable(costs['initialization']):
        p = ledger.restart(backend.restart, static=True)
        return p, {'status': 'STATIC_FALLBACK', 'operations': ledger.rows, **diagnostics}
    ledger.pay('initialization', costs['initialization'], backend.initialize)
    for d in range(depth):
        # Combined affordability prevents paying for sensing if scoring cannot follow.
        if not ledger.affordable(costs['cheap_measurement'] + costs['score']):
            p = ledger.restart(backend.restart)
            return p, {'status': 'LATE_FALLBACK', 'operations': ledger.rows, **diagnostics}
        ledger.pay('cheap_measurement', costs['cheap_measurement'], lambda: backend.measure(d), depth=d)
        score = ledger.pay('score', costs['score'], lambda: backend.score(d), depth=d)
        supported = list(score['supported'])
        if (any(type(a) is not int or not 0 <= a < 10 for a in supported)
                or len(set(supported)) != len(supported)):
            raise ValueError('Invalid E6 support set')
        recovery = action_id(score['recovery'], 'Recovery action')
        if recovery not in (8, 9) and recovery not in supported:
            raise ValueError('Unregistered recovery action')
        action = recovery
        origin = 'recovery_no_support'
        if supported:
            cert = certificate(d, dict(score)) if certificate else None
            if cert is not None and certified_skip(score, supported, cert):
                action = score['action']
                origin = 'certified_skip'
                diagnostics['certified_skips'] += 1
            elif ledger.affordable(costs['acquisition'] + costs['action']):
                action = action_id(ledger.pay('acquisition', costs['acquisition'],
                                             lambda: backend.acquire(d), depth=d), 'Acquired action')
                origin = 'acquired_policy'
                diagnostics['acquisitions'] += 1
            else:
                origin = 'recovery_unaffordable_acquisition'
                diagnostics['unaffordable_acquisitions'] += 1
                diagnostics['recoveries'] += 1
        else:
            diagnostics['recoveries'] += 1
        try:
            ledger.pay('action', costs['action'], lambda: backend.act(action, d), depth=d)
            ledger.rows[-1].update(action_id=action, supported_actions=supported, origin=origin,
                                   member_index=score.get('member_index'),local_depth=score.get('local_depth',d))
        except Unaffordable:
            p = ledger.restart(backend.restart)
            return p, {'status': 'LATE_FALLBACK', 'operations': ledger.rows, **diagnostics}
    p = ledger.pay('prediction', costs['prediction'], backend.predict, terminal=True, outcome='completed', depth=depth)
    return p, {'status': 'COMPLETED', 'operations': ledger.rows, **diagnostics}


def completion_requirement(trace):
    """Necessary terminal cap from an actually completed operation trace."""
    from .protocol import terminal_reserve_requirement
    rows = trace.get('operations', [])
    if not rows or rows[-1].get('outcome') != 'completed':
        return {'status': 'NOT_COMPLETED', 'necessary_cap': None,
                'reason': 'a successful controller trajectory was not observed'}
    if rows[-1].get('operation') != 'prediction':
        raise ValueError('Completion must end in prediction')
    return {'status': 'COMPLETED', **terminal_reserve_requirement(
        float(rows[-1]['cumulative_work']), float(rows[-1]['paid_work']),
        float(rows[-1]['comparator_reserve']))}
