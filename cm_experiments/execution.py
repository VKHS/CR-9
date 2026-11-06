"""Label-free E3--E6 execution on the released frozen graph predictors.

The nine original route implementations are left unchanged. Additional arms
have their own source identity, traces and experiment directory. The work unit
is the existing product counter, not a time/energy guarantee or complete FLOPs.
"""
from __future__ import annotations

import copy
import math
import time
import numpy as np
import torch
from torch import nn

from cross_morse.common import precision_context, synchronize, jsonable
from cross_morse.core import ACTIONS, site_transport_accounting, site_action_summary
from cross_morse.exploration import sensor_bank
from cross_morse.policies import beam_plan, execute_policy, recovery_action, PlanningBudgetExceeded
from cross_morse.recovery import label_free_safe
from cross_morse.tasks import probabilities, task_spec
from .budget import run_budget_episode


def initial_features(h0, graph):
    """Mean-pooled H0 plus log size, edge count, mean and max weighted degree."""
    if not torch.isfinite(h0).all():
        raise FloatingPointError('Nonfinite initial representation')
    degree = graph.degree.float()
    stats = h0.new_tensor([math.log1p(graph.num_nodes), math.log1p(graph.num_edges),
                          math.log1p(float(degree.mean())), math.log1p(float(degree.max()))], dtype=torch.float32)
    return torch.cat([h0.float().mean(0), stats]).cpu().numpy()


class FrozenEpisode:
    def __init__(self, model, x, graph, config, controller, candidate, *, comparator=None,
                 comparator_controller=None, reference=None, cheap_controller=None):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Arm execution requires a frozen eval-mode predictor')
        self.model, self.x, self.graph, self.config = model, x, graph, config
        self.controller, self.candidate = controller, candidate
        self.cheap_controller = cheap_controller
        self.cheap_state = None
        self.comparator = comparator
        self.comparator_controller = comparator_controller if comparator_controller is not None else controller
        self.reference = reference
        self.L = int(config['model']['depth'])
        self.h0 = self.h = self.last_safe = None
        self.state = None; self.mass = self.hops = 0.; self.previous = -1
        self.banks = {}; self.actions = []; self.origins = []; self.safe_depths = []
        self.decisions = []; self.last_safe_depth = 0; self.measurement_depths = []
        self.search = {'calls': 0}; self.last_score = None
        self.model.bind_graph(graph)
        graph.reset_counters(); model.reset_counters(); controller.reset_counters()
        if cheap_controller is not None and cheap_controller is not controller:
            cheap_controller.reset_counters()

    def products(self):
        return (self.graph.sparse_multiply_adds + self.model.counters['dense_multiply_adds']
                + self.controller.counters.get('approx_dense_multiply_adds', 0)
                + (self.cheap_controller.counters.get('approx_dense_multiply_adds',0)
                   if self.cheap_controller is not None and self.cheap_controller is not self.controller else 0)
                + sum(b.counters.get('dense_multiply_adds', 0) for b in self.banks.values()))

    def measured(self, callback):
        before = self.products(); value = callback(); work = self.products() - before
        if work < 0:
            raise RuntimeError('Operation reset work counters inside a measured section')
        return value, float(work)

    def _bank(self, profile):
        if profile not in self.banks:
            bank = sensor_bank(self.graph, self.model, self.config, profile)
            bank.prepare(self.h0, self.model.logits(self.h0))
            self.banks[profile] = bank
        return self.banks[profile]

    def _measure(self, depth, profile):
        bank = self._bank(profile)
        y = bank.measure(self.h, self.model.logits(self.h), depth, self.L, self.mass, self.hops).cpu().numpy()
        self.measurement_depths.append({'depth': depth, 'profile': profile})
        return y

    def initialize(self):
        def operation():
            self.h0 = self.model.encode(self.x); self.h = self.h0; self.last_safe = self.h0
            return None
        return self.measured(operation)

    def initial_state(self, profile='full'):
        y = self._measure(0, profile)
        self.state = self.controller.start(y)
        return y

    def measure(self, depth):
        """Update a separately fitted Lean observer; no full sensor is hidden."""
        def operation():
            if self.cheap_controller is None:
                raise ValueError('E6 requires the separately fitted Lean controller')
            y = self._measure(depth, 'lean')
            self.cheap_y = y
            self.cheap_state = (self.cheap_controller.start(y) if self.cheap_state is None else
                                self.cheap_controller.observe(self.cheap_state,y,depth,self.previous))
            if label_free_safe(y):
                self.last_safe = self.h.clone(); self.last_safe_depth = depth
            return y
        return self.measured(operation)

    def score(self, depth):
        def operation():
            if self.cheap_state is None:
                raise ValueError('Cheap measurements must precede E6 scoring')
            control = self.cheap_controller
            supported = control.supported(self.cheap_state.y, depth)
            scored = control.score(self.cheap_state, supported, kind='radius',
                      kappa=self.candidate.get('kappa',1.),lambda_rho=self.config['controller']['lambda_rho']) if supported else {}
            finite = [a for a in supported if np.isfinite(scored[a]['score'])]
            fallback = recovery_action(supported)
            winner = min(finite, key=lambda a:(scored[a]['score'], a)) if finite else fallback
            values = sorted(scored[a]['score'] for a in finite)
            gap = values[1] - values[0] if len(values)>1 else (math.inf if values else 0.)
            self.last_score = {'supported': list(map(int,supported)), 'action': int(winner),
                               'recovery': int(fallback), 'gap':gap,
                               'state_depth': self.cheap_state.depth,
                               'score_origin':'separately fitted Lean observer and critic'}
            return self.last_score
        return self.measured(operation)

    def acquire(self, depth):
        def operation():
            y = self._measure(depth, self.candidate.get('profile', 'full'))
            if self.state is None:
                self.state = self.controller.start(y, self.previous)
                self.state.depth = depth
            else:
                self.state = self.controller.observe(self.state, y, depth, self.previous)
            supported = self.controller.supported(y, depth)
            if not supported or not self.state.valid:
                return recovery_action(supported)
            if self.candidate['route'] == 'mpc_rh':
                try:
                    plan = beam_plan(self.controller, self.state,
                                     min(int(self.candidate.get('horizon',1)), self.L-depth), self.config, self.search)
                    return plan[0] if plan else recovery_action(supported)
                except PlanningBudgetExceeded:
                    return recovery_action(supported)
            scores = self.controller.score(self.state, supported, kind=self.candidate['route'],
                         kappa=self.candidate.get('kappa',1.), lambda_rho=self.config['controller']['lambda_rho'])
            finite = [a for a in supported if np.isfinite(scores[a]['score'])]
            return min(finite, key=lambda a:(scores[a]['score'],a)) if finite else recovery_action(supported)
        return self.measured(operation)

    def act(self, action, depth, *, predict_state=True):
        def operation():
            if action not in range(10):
                raise ValueError('Action outside registry')
            self.h = self.model.step(self.h, self.h0, self.graph, action, self.last_safe, site=depth)
            if not torch.isfinite(self.h).all():
                raise FloatingPointError('Nonfinite frozen plant state')
            account = site_transport_accounting(action, depth, self.model)
            self.mass += account['mass']; self.hops += account['hops']
            self.actions.append(int(action)); self.safe_depths.append(int(self.last_safe_depth)); self.previous = action
            if predict_state and self.state is not None and depth+1 < self.L:
                self.state = self.controller.predict(self.state, action)
            return None
        return self.measured(operation)

    def predict(self):
        def operation():
            p = probabilities(self.model.logits(self.h).float(), getattr(self.model,'task',task_spec(self.config))).cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError('Nonfinite final probabilities')
            return p
        return self.measured(operation)

    def restart(self):
        if self.comparator is None:
            raise ValueError('An executable comparator is required for E6')
        p, trace = execute_policy(self.model,self.x,self.graph,self.config,self.comparator,
                                 self.comparator_controller,self.reference)
        self.comparator_trace = trace
        # Actual restart in the same DEPLOYMENT product coordinate as qC.
        # Qualification is separately executed/logged by the reference executor;
        # it does not silently inflate only the restart side of the comparison.
        from .protocol import deployed_work
        return p, deployed_work(trace)

    def trace(self, arm, *, extra=None):
        return jsonable({'arm':arm,'actions':self.actions,'safe_depths':self.safe_depths,
                         'measurements':self.measurement_depths,'model_calls':dict(self.controller.counters),
                         'cheap_model_calls':dict(self.cheap_controller.counters) if self.cheap_controller is not None else {},
                         'search_calls':self.search['calls'],'multiply_add_work':self.products(),
                         'decision_trace':self.decisions,'labels_used_for_execution':False,
                         'work_scope':'instrumented multiply-add products; reductions, memory transfers and elapsed time are separate',
                         **site_action_summary(self.actions,self.model), **(extra or {})})


def operation_bounds(episode):
    """Conservative static bounds for the released eager implementation.

    Counts cover its sparse/dense-product counters, not all hardware operations.
    Conservative bounds may make E6 infeasible at a tight ceiling; that outcome
    is reported, not silently relaxed. A tighter externally calibrated bound
    file can be supplied and must survive runtime bound checks.
    """
    m,g,c = episode.model,episode.graph,episode.config
    n,d,e = g.num_nodes,m.hidden_dim,g.num_edges
    weights = sum(p.numel() for p in m.parameters())
    dense = n*weights + 2*n*n*d*max(1,len(getattr(m,'blocks',[])))
    sensor = 2*n*max(d,32)**2 + (3*c['sensor']['degree']+4)*e*d + 2*n*weights
    s = episode.controller.observer.dimension
    nets = [episode.controller.critic,episode.controller.transition_model,episode.controller.stage_model,
            episode.controller.terminal_model]
    ctrl = 10*sum(p.numel() for net in nets for p in net.parameters()) + 8*s**3 + 100*s
    horizon = min(int(episode.candidate.get('horizon',1)),episode.L)
    calls = horizon*max(1,c['controller']['beam_width'])*(1+3*max(1,c['controller']['top_k'])) + 4
    initialization = dense
    action = dense + 8*e*d + ctrl
    acquisition = sensor + calls*ctrl
    prediction = n*d*m.num_classes
    # Includes qualification sensing if the comparator executor performs it.
    comparator = initialization + sensor + episode.L*(action + sensor + acquisition) + prediction
    return {k:float(v) for k,v in dict(initialization=initialization,
                 cheap_measurement=sensor,score=ctrl,acquisition=acquisition,
                 action=action,prediction=prediction,comparator=comparator).items()}


@torch.no_grad()
def execute_arm(arm, model, x, graph, config, controller, candidate, *, decoder=None,
                comparator=None, comparator_controller=None, reference=None, budget=None,
                costs=None, certificate=None, cheap_controller=None):
    if arm not in ('E3','E4','E5','E6'):
        raise ValueError('Unknown budget-matched arm')
    if candidate.get('route') not in ('one_step','radius','mpc_rh'):
        raise ValueError('A frozen feedback teacher configuration is required')
    ep = FrozenEpisode(model,x,graph,config,controller,candidate,comparator=comparator,
                       comparator_controller=comparator_controller,reference=reference,cheap_controller=cheap_controller)
    synchronize(x.device); start=time.perf_counter()
    with precision_context(config,x.device):
        if arm=='E6':
            if budget is None or not np.isfinite(budget) or budget<0:
                raise ValueError('E6 needs a finite, predeclared work budget')
            bounds=operation_bounds(ep) if costs is None else dict(costs)
            p, ledger=run_budget_episode(ep,depth=ep.L,budget=budget,costs=bounds,certificate=certificate)
            trace=ep.trace(arm,extra=ledger)
            trace['deployed_multiply_add_work']=sum(row['paid_work'] for row in ledger['operations'])
            trace['qualification_multiply_add_work']=float(getattr(ep,'comparator_trace',{}).get('qualification_multiply_add_work',0.))
            trace['multiply_add_work']=trace['deployed_multiply_add_work']+trace['qualification_multiply_add_work']
            trace['cost_bounds']=bounds
            trace['hard_cap_scope']='declared multiply-add work bounds; no latency/energy guarantee'
            if hasattr(ep,'comparator_trace'):
                trace['comparator_restart_trace']=ep.comparator_trace
        else:
            ep.initialize()
            if arm in ('E3','E4'):
                if decoder is None or decoder.spec.max_depth!=ep.L:
                    raise ValueError('E3/E4 require a matching trained initial-input decoder')
                ep.initial_state(candidate.get('profile','full'))
                features=initial_features(ep.h0,graph)
                # All support filtering is on initial/model-predicted states.
                # No controlled plant step has occurred during decoding.
                mask_state=ep.state
                def mask(depth,prefix):
                    nonlocal mask_state
                    if depth:
                        mask_state=controller.predict(mask_state,int(prefix[-1]))
                    return [int(a) for a in controller.supported(mask_state.y,depth) if int(a)!=9]
                sequence,decode_trace=decoder.decode(features,mask)
                for depth,action in enumerate(sequence):
                    ep.act(action,depth,predict_state=False)
                p,_=ep.predict()
                trace=ep.trace(arm,extra=decode_trace)
                trace['multiply_add_work']+=decode_trace['decoder_multiply_adds']
                trace['support_scope']='initial-state/model-predicted mask; actual-state support requires separate offline audit'
            else:
                ep.initial_state(candidate.get('profile','full'))
                for depth in range(ep.L):
                    # Only model-predicted states enter planning after depth zero.
                    supported=controller.supported(ep.state.y,depth)
                    if label_free_safe(ep.state.y):
                        ep.last_safe=ep.h.clone();ep.last_safe_depth=depth
                    try:
                        plan=beam_plan(controller,ep.state,min(int(candidate.get('horizon',1)),ep.L-depth),config,ep.search)
                        action=int(plan[0]) if plan else recovery_action(supported)
                    except PlanningBudgetExceeded:
                        plan=[];action=recovery_action(supported)
                    ep.decisions.append({'depth':depth,'planned_sequence':plan,'action':action,
                                         'state_source':'model_prediction' if depth else 'initial_measurement'})
                    ep.act(action,depth)
                p,_=ep.predict()
                trace=ep.trace(arm,extra={'intermediate_observations_used':0,
                     'planning_mode':'replan on model states at each depth',
                     'call_matching':'same horizon, beam and top-k; actual calls reported, not asserted equal',
                     'a9_anchor_rule':'save actual representation when model-predicted measurements satisfy fixed safe criterion'})
    synchronize(x.device);trace['wall_seconds']=time.perf_counter()-start
    trace['timing_scope']='CUDA-synchronized whole episode; includes output copy and controller instrumentation'
    return p,trace


class EnsembleEpisode:
    """One restart reserve for the complete ensemble, not independent fallbacks.

    Members execute serially. Every controller operation is paid under the same
    ledger. No output is returned until all members finish, or ALL comparator
    members have been reexecuted after an abort. Predictions are formed last.
    """
    def __init__(self, members):
        if not members or len({m.L for m in members}) != 1:
            raise ValueError('Nonempty ensemble with a common controlled depth required')
        self.members=members; self.L=members[0].L; self.restart_traces=[]

    def _member(self,depth):
        if not 0<=depth<len(self.members)*self.L:raise ValueError('Ensemble depth outside range')
        return self.members[depth//self.L], depth%self.L

    def initialize(self):
        work=0.
        for m in self.members:
            _,w=m.initialize();work+=w
        return None,work

    def measure(self,depth):
        member,d=self._member(depth);return member.measure(d)

    def score(self,depth):
        member,d=self._member(depth);score,work=member.score(d)
        return dict(score,member_index=depth//self.L,local_depth=d),work

    def acquire(self,depth):
        member,d=self._member(depth);return member.acquire(d)

    def act(self,action,depth):
        member,d=self._member(depth);return member.act(action,d)

    def predict(self):
        values=[];work=0.
        for m in self.members:
            p,w=m.predict();values.append(p);work+=w
        return np.mean(np.stack(values).astype(np.float64),axis=0).astype(np.float32),work

    def restart(self):
        values=[];work=0.
        for m in self.members:
            p,w=m.restart();values.append(p);work+=w
            self.restart_traces.append(m.comparator_trace)
        return np.mean(np.stack(values).astype(np.float64),axis=0).astype(np.float32),work


@torch.no_grad()
def execute_budget_ensemble(models, x, graph, config, controllers, cheap_controllers,
                            candidate, comparator, comparator_controllers, reference,
                            *, budget, per_member_costs=None, certificate=None):
    count=len(models)
    if not count or any(len(v)!=count for v in (controllers,cheap_controllers,comparator_controllers)):
        raise ValueError('Ensemble checkpoint and controller counts must match')
    if per_member_costs is not None and len(per_member_costs)!=count:
        raise ValueError('One frozen cost specification is required per ensemble member')
    episodes=[FrozenEpisode(m,x,graph,config,c,candidate,comparator=comparator,
               comparator_controller=qc,reference=reference,cheap_controller=lean)
              for m,c,lean,qc in zip(models,controllers,cheap_controllers,comparator_controllers)]
    costs=[operation_bounds(ep) for ep in episodes] if per_member_costs is None else per_member_costs
    # Initialization and final predictions cover every member; an inner step
    # executes exactly one member. The largest per-member bound is conservative.
    bounds={k:float(sum(v[k] for v in costs) if k in ('initialization','prediction','comparator')
              else max(v[k] for v in costs)) for k in costs[0]}
    backend=EnsembleEpisode(episodes)
    synchronize(x.device);start=time.perf_counter()
    with precision_context(config,x.device):
        p,trace=run_budget_episode(backend,depth=episodes[0].L*count,budget=budget,
                                  costs=bounds,certificate=certificate)
    synchronize(x.device)
    trace.update({'arm':'E6','ensemble_size':count,'wall_seconds':time.perf_counter()-start,
        'deployed_multiply_add_work':sum(row['paid_work'] for row in trace['operations']),
        'qualification_multiply_add_work':sum(float(t.get('qualification_multiply_add_work',0.)) for t in backend.restart_traces),
        'multiply_add_work':sum(row['paid_work'] for row in trace['operations'])+
            sum(float(t.get('qualification_multiply_add_work',0.)) for t in backend.restart_traces),
        'hard_cap_scope':'instrumented multiply-add products under predeclared bounds; not latency or energy',
        'budget_scope':'whole ensemble, one reserve, whole-comparator restart',
        'cost_bounds':bounds,'per_member_bounds':costs,
        'member_traces':[ep.trace('E6') for ep in episodes],
        'comparator_restart_traces':backend.restart_traces,
        'labels_used_for_execution':False,
        'timing_scope':'synchronized instrumented whole ensemble'})
    return p,jsonable(trace)


@torch.no_grad()
def audit_sequence(model,x,graph,config,controller,actions,safe_depths):
    """Offline actual-state support audit; never revise the open-loop decisions.

    This separate replay charges its entire work as qualification. It is not
    hidden in deployed E3/E4/E5 inference and its observations are never fed to
    those decision functions. A9 anchors are replayed exactly from the trace.
    """
    if len(actions)!=len(safe_depths) or not actions:
        raise ValueError('Complete action and reset-anchor trace required')
    graph.reset_counters();model.reset_counters();controller.reset_counters()
    model.bind_graph(graph);violations=[];synchronize(x.device);started=time.perf_counter()
    with precision_context(config,x.device):
        h0=model.encode(x);h=h0;states={0:h0};mass=hops=0.
        bank=sensor_bank(graph,model,config,controller.observer.profile)
        bank.prepare(h0,model.logits(h0))
        for d,a in enumerate(actions):
            y=bank.measure(h,model.logits(h),d,config['model']['depth'],mass,hops).cpu().numpy()
            supported=controller.supported(y,d)
            if a not in (8,9) and a not in supported:
                violations.append({'depth':d,'action':int(a),'supported':list(map(int,supported))})
            anchor=int(safe_depths[d])
            if anchor not in states:raise ValueError('Reset anchor is not in the executed past')
            h=model.step(h,h0,graph,int(a),states[anchor],site=d);states[d+1]=h
            item=site_transport_accounting(int(a),d,model);mass+=item['mass'];hops+=item['hops']
        p=probabilities(model.logits(h).float(),getattr(model,'task',task_spec(config))).cpu().numpy()
    synchronize(x.device)
    return p,{'feasible':not violations,'support_violations':violations,'decision_intervention':False,
              'qualification_multiply_add_work':int(graph.sparse_multiply_adds+model.counters['dense_multiply_adds']+
                controller.counters['approx_dense_multiply_adds']+bank.counters['dense_multiply_adds']),
              'wall_seconds':time.perf_counter()-started,
              'scope':'separate replay on the fixed graph; whole audit work excluded from deployed work and reported independently'}
