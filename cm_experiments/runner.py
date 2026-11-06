"""Development-selected E3--E6 execution against an immutable confirmed run.

The parent predictor and its scientific decisions are never modified. Decoder
fitting and arm configuration selection have disjoint Dexplore observations.
D_compare is opened only after the selection record is written and hashed.
The feedback teacher was chosen on the separate D_confirm role.
"""
from __future__ import annotations
import copy
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from cross_morse.common import precision_context, jsonable
from cross_morse.data import load_data
from cross_morse.provenance import verify_stage_chain, current_source_sha256, sha256_state_dict
from cross_morse.training import load_plant
from cross_morse.learning import Controller
from cross_morse.runtime import graph_operator, is_graph_collection
from cross_morse.policies import execute_policy
from cross_morse.evaluation import score_observations
from cross_morse.tasks import sample_losses, task_spec
from .artifacts import write_json, sha256, canonical, fresh_output, source_identity, seal
from .schedules import fit_decoder, save_decoder
from .accounting import summarize_work
from .execution import initial_features
from .protocol import budget_contract, choose_development, deployed_work, candidate_id
from .development import development_partition, subset_labels, open_loop_family
from .study_execution import StudyBackend, variants
from .holdout import ARM_SCHEMA, comparison_contract, require_arm_contract


def _read(path):
    return json.loads(Path(path).read_text())


def select_policies(root, config, allow_fixture=False):
    candidates = _read(root/'select/candidates.json')
    decision = _read(root/'confirm/decision.json')
    teacher = copy.deepcopy(candidates[decision['confirmed_index']])
    comparator = copy.deepcopy(candidates[decision['comparator_index']])
    override = False
    if teacher['route'] not in ('one_step', 'radius', 'mpc_rh'):
        if not allow_fixture:
            raise ValueError('Confirmed policy is open-loop; this feedback comparison is not applicable')
        teacher = copy.deepcopy(next(c for c in candidates if c['route'] == 'mpc_rh'
                                     and c['ensemble'] == comparator['ensemble']))
        override = True
    return teacher, comparator, {'teacher': teacher, 'comparator': comparator,
                                 'fixture_teacher_override': override}


def episodes(data, role, max_graphs=0):
    if is_graph_collection(data):
        ids = list(map(int, data.graph_splits[role]))
        if max_graphs:
            ids = ids[:max_graphs]
        for i in ids:
            yield i, data.graph(i)
    else:
        yield 0, data


def _sequence_candidate(sequence, profile):
    return {'route': 'fixed_sequence', 'profile': profile, 'depth': len(sequence),
            'sequence': list(sequence), 'ensemble': 1, 'behavior_id': '00000000'}


def collect_targets(root, data, config, teacher, comparator, partition, *, device, seeds,
                    proposal_limit, output):
    family, coverage = open_loop_family(root, comparator, depth=config['model']['depth'],
                                        proposal_limit=proposal_limit)
    profile = teacher.get('profile', 'full')
    control = Controller.load(root/f'fit/{profile}/controller.pt', 'cpu')
    critic_params = sum(p.numel() for p in control.critic.parameters())
    reference = dict(np.load(root/'explore/nominal_reference.npz', allow_pickle=False))
    features, oracle, teachers, lengths, teacher_lengths, groups, members, eids, detail = ([] for _ in range(9))
    L = int(config['model']['depth'])
    fit_rows = [r for r in partition['episodes'] if r['fit_ids']]
    for seed in seeds:
        model = load_plant(data, config, root/'train', seed, device)
        before = sha256_state_dict(model)
        for row in fit_rows:
            eid = row['episode_id']; episode = data.graph(eid) if is_graph_collection(data) else data
            graph = graph_operator(episode, device); x = episode.x.to(device)
            with torch.no_grad(), precision_context(config, torch.device(device)):
                model.bind_graph(graph); h0 = model.encode(x); feat = initial_features(h0, graph)
            _, teacher_trace = execute_policy(model, x, graph, config, teacher, control, reference)
            teacher_seq = list(map(int, teacher_trace['actions']))
            if not 1 <= len(teacher_seq) <= L:
                raise ValueError('Teacher did not produce a valid terminal depth')
            normalized = [8 if a == 9 else a for a in teacher_seq]
            proposals, predictions, qualification = [], [], []
            # The actual policy is run, so MPC-OL and learned-stop depend on H0
            # and preserve their own planning/decoding cost and stopping depth.
            for item in family:
                candidate = item['candidate']
                cp = candidate.get('profile', 'full')
                ctl = Controller.load(root/f"fit/{'full' if cp == 'none' else cp}/controller.pt", 'cpu')
                p, trace = execute_policy(model, x, graph, config, candidate, ctl, reference)
                proposals.append({'candidate_id': item['configuration_id'], 'route': candidate['route'],
                                  'origin': item['source'], 'actions': trace['actions'],
                                  'terminal_depth': trace['terminal_depth']})
                predictions.append(p)
                qualification.append({'feasible': trace['feasible'], 'support_violations': trace['support_violations'],
                                      'deployed_work': deployed_work(trace), 'total_work': trace['multiply_add_work']})
            for origin, seq in [('source_reset', [8]*L), ('normalized_feedback_teacher', normalized)]:
                candidate = _sequence_candidate(seq, profile)
                p, trace = execute_policy(model, x, graph, config, candidate, control, reference)
                proposals.append({'candidate_id': canonical(candidate)[:24], 'route': 'fixed_sequence',
                                  'origin': origin, 'actions': trace['actions'], 'terminal_depth': len(seq)})
                predictions.append(p)
                qualification.append({'feasible': trace['feasible'], 'support_violations': trace['support_violations'],
                                      'deployed_work': deployed_work(trace), 'total_work': trace['multiply_add_work']})
            labels = subset_labels(data, episode, 'explore', row['fit_ids'])
            ids = np.asarray(row['fit_ids'], dtype=np.int64)
            risk = [float(np.mean(sample_losses(p[ids], labels, task_spec(config, episode.metadata))))
                    for p in predictions]
            if not np.isfinite(risk).all():
                raise ValueError('Nonfinite oracle loss')
            eligible = [j for j, q in enumerate(qualification) if q['feasible']]
            if not eligible:
                raise ValueError('No support-qualified development proposal')
            winner = min(eligible, key=lambda j: (risk[j], proposals[j]['candidate_id']))
            chosen = list(map(int, proposals[winner]['actions']))
            features.append(feat); oracle.append(chosen + [0]*(L-len(chosen)))
            teachers.append(normalized + [0]*(L-len(normalized)))
            lengths.append(len(chosen)); teacher_lengths.append(len(normalized))
            groups.append(f'graph:{eid}' if is_graph_collection(data) else f'checkpoint:{seed}')
            members.append(seed); eids.append(eid)
            detail.append({'seed': seed, 'episode_id': eid, 'proposals': proposals, 'losses': risk,
                           'support_audits': qualification, 'selected_proposal': winner,
                           'teacher_actions': teacher_seq, 'teacher_safe_depths': teacher_trace['safe_depths'],
                           'a9_source_reset_conversions': teacher_seq.count(9)})
        if sha256_state_dict(model) != before:
            raise RuntimeError('Frozen predictor changed during target collection')
    if not features:
        raise ValueError('No fitting episodes')
    arrays = {'initial_features': np.stack(features), 'targets_E3': np.asarray(oracle, dtype=np.int64),
              'targets_E4': np.asarray(teachers, dtype=np.int64), 'lengths_E3': np.asarray(lengths, dtype=np.int64),
              'lengths_E4': np.asarray(teacher_lengths, dtype=np.int64), 'group_ids': np.asarray(groups),
              'source_seeds': np.asarray(members, dtype=np.int64), 'episode_ids': np.asarray(eids, dtype=np.int64),
              'roles': np.full(len(features), 'Dexplore')}
    np.savez_compressed(output/'development_targets.npz', **arrays)
    write_json(output/'development_oracle.json', {'records': detail, 'family_coverage': coverage,
        'scope': 'finite whole-family development oracle, not a global supported-sequence optimum',
        'labels': 'Dexplore fitting subset only', 'terminal_depths': 'preserved with a decoder STOP symbol',
        'grouping': partition['grouping'], 'unique_input_contexts': partition['unique_input_contexts']})
    return arrays, critic_params, family


def _write_environment(out, device):
    write_json(out/'environment.json', {'schema': 'cm-arm-observed-environment-1',
        'python_version': platform.python_version(), 'architecture': platform.machine(),
        'operating_system': platform.system(), 'device_requested': str(device),
        'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
        'visible_gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                                if d.metadata.get('Name'))),
        'slurm': {k: os.environ[k] for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID') if k in os.environ},
        'scope': 'captured for this invocation, not the parent environment'})


def select_development(backend, data, partition, registry, contract, output):
    rows = {a: [] for a in registry}
    contexts = [r for r in partition['episodes'] if r['selection_ids']]
    reference = {}
    for row in contexts:
        ep = data.graph(row['episode_id']) if is_graph_collection(data) else data
        reference[row['episode_id']] = backend.execute(ep, {'kind': 'policy', 'candidate': backend.comparator})
    for arm, candidates in registry.items():
        for spec in candidates:
            observed, risks, work, refs = [], [], [], []
            raw_predictions, raw_labels, raw_ids, raw_units, raw_inputs = [], [], [], [], []
            qualified = True; executed = True
            for row in contexts:
                eid = row['episode_id']; ep = data.graph(eid) if is_graph_collection(data) else data
                qc = reference[eid]
                result = backend.execute(ep, spec, qc_reference_work=qc['deployed_work'],
                                         hard_cap_ratio=contract.hard_cap_ratio)
                entry = {'episode_id': eid, 'status': result['status'], 'work': result['deployed_work'],
                         'qualification_work': result['embedded_qualification_work'] + result['separate_qualification_work'],
                         'qc_work': qc['deployed_work'], 'support_qualified': result['support_qualified'],
                         'traces': result['traces']}
                if result['prediction'] is None:
                    executed = False; entry['loss'] = None
                else:
                    labels = subset_labels(data, ep, 'explore', row['selection_ids'])
                    ids = np.asarray(row['selection_ids'], dtype=np.int64)
                    risk = float(np.mean(sample_losses(result['prediction'][ids], labels,
                                                       task_spec(backend.config, ep.metadata))))
                    if not np.isfinite(risk):
                        raise ValueError('Nonfinite development-selection loss')
                    entry['loss'] = risk; risks.append(risk)
                    raw_predictions.append(result['prediction'][ids]); raw_labels.append(labels)
                    gids = ep.global_sample_ids
                    raw_ids.extend(ids.tolist() if gids is None else gids[ids].tolist())
                    raw_units.extend(ep.cluster_ids[ids].tolist()); raw_inputs.extend([eid]*len(ids))
                qualified = qualified and result['support_qualified']
                work.append(result['deployed_work']); refs.append(qc['deployed_work']); observed.append(entry)
            pred_path = f"development_predictions/{arm}/{spec['configuration_id']}.npz"
            if executed:
                dest = output/pred_path; dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, y_pred=np.concatenate(raw_predictions), y_true=np.concatenate(raw_labels),
                                    observation_id=np.asarray(raw_ids), unit_id=np.asarray(raw_units),
                                    input_id=np.asarray(raw_inputs))
            rows[arm].append({'predictions': pred_path if executed else None, 'configuration_id': spec['configuration_id'], 'role': 'Dexplore_selection',
                              'status': 'EXECUTED' if executed else 'INFEASIBLE',
                              'support_qualified': qualified, 'loss': float(np.mean(risks)) if executed else None,
                              'mean_work': float(np.mean(work)), 'mean_qc_work': float(np.mean(refs)),
                              'work_ratio': float(np.mean(work)/np.mean(refs)), 'inputs': len(contexts),
                              'observations': observed})
    selected = {a: choose_development(v, contract) for a, v in rows.items()}
    result = {'schema': 'cm-arm-development-selection-1', 'budget': contract.record(),
              'partition_sha256': canonical(partition), 'registry_sha256': canonical(registry),
              'role': 'Dexplore_selection', 'loss_weighting': 'equal input mean of node/task loss',
              'comparison_used': False, 'candidates': rows, 'decisions': selected}
    write_json(output/'development_selection.json', jsonable(result))
    return selected


def run_study(base_run, output, *, device='cpu', trust=False, allow_fixture=False, allow_posthoc=False,
              max_graphs=0, proposal_limit=0, epochs=100, patience=15, training_seed=17,
              budget_ratio=None, hard_cap_ratio=None, costs_path=None, selection_fraction=.25,
              learning_rates=(.001, .0003), budget_tolerance=.01):
    if not trust:
        raise ValueError('Explicit trust is required before loading local model checkpoints')
    root = Path(base_run).resolve(); config = _read(root/'config.json')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    fixture = config.get('dataset') == 'synthetic'
    if fixture and not allow_fixture:
        raise ValueError('Generated test input requires --allow-fixture')
    if config['model'].get('precision') == 'bf16' and not str(device).startswith('cuda'):
        raise ValueError('BF16 execution requires CUDA; no implicit CPU replacement')
    later = any((root/p).exists() for p in ('admit/decision.json', 'final/results_summary.json'))
    if later and not allow_posthoc and not allow_fixture:
        raise ValueError('Parent has admission/test decisions; use a confirmed pre-admission parent or declare --allow-posthoc')
    if type(max_graphs) is not int or max_graphs < 0 or type(proposal_limit) is not int or proposal_limit < 0:
        raise ValueError('Invalid graph/proposal safety ceilings')
    rates = list(map(float, learning_rates))
    if not rates or len(rates) != len(set(rates)) or any(not np.isfinite(v) or v <= 0 for v in rates):
        raise ValueError('Predeclare distinct positive learning rates')
    dataset = config.get('benchmark_dataset', config['dataset'])
    contract = budget_contract(dataset, expected=budget_ratio, hard_cap=hard_cap_ratio,
                               tolerance=budget_tolerance, fixture=fixture)
    data = load_data(root).protect_labels()
    holdout = comparison_contract(data, root)
    verify_stage_chain(root, 'confirm', expected_source_sha256=current_source_sha256())
    teacher, comparator, policy_record = select_policies(root, config, allow_fixture)
    if teacher['ensemble'] != comparator['ensemble']:
        raise ValueError('This matched-ensemble study requires equal qC/qM member counts')
    seeds = list(map(int, config['training']['seeds']))
    partition = development_partition(data, seed=training_seed, max_graphs=max_graphs,
                                      selection_fraction=selection_fraction)
    out = fresh_output(output, root); source = Path(__file__).resolve().parents[1]
    identity = source_identity(source)
    exp = {'schema': ARM_SCHEMA, 'dataset': dataset, 'input_dataset': config['dataset'],
           'profile': config['profile'], 'source_runtime_sha256': current_source_sha256(), 'arm_source': identity,
           'base_configuration_sha256': canonical(config),
           'base_confirm_manifest_sha256': sha256(root/'provenance/stages/05_confirm.json'),
           'comparison_holdout': holdout,
           'training_seeds': seeds, 'ensemble_members': seeds[:teacher['ensemble']], 'training_seed': training_seed,
           'training_role': 'Dexplore_fitting', 'selection_role': 'Dexplore_selection', 'evaluation_role': 'D_compare',
           'budget': contract.record(), 'budget_ratio': contract.expected_ratio,
           'protocol': 'posthoc' if later else 'pre_admission', 'prior_admission_or_test_available': later,
           'max_graphs': max_graphs, 'proposal_limit': proposal_limit, 'epochs': epochs, 'patience': patience,
           'learning_rates': rates, 'publication_deposit': False,
           'scope': 'one real execution of the registered study, not the submitted S46 historical implementation or a full campaign',
           'input_scope': partition['grouping'], 'unique_input_contexts': partition['unique_input_contexts'],
           'created_utc': datetime.now(timezone.utc).isoformat(), **policy_record}
    write_json(out/'experiment.json', exp); write_json(out/'base_config.json', config)
    write_json(out/'development_partition.json', partition); _write_environment(out, device)
    np.savez_compressed(out/'data_roles.npz', **{k: np.asarray(v) for k, v in data.splits.items()})
    for rel in identity['files']:
        dest = out/'code_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source/rel, dest)
    arr, critic_count, family = collect_targets(root, data, config, teacher, comparator, partition,
        device=device, seeds=seeds, proposal_limit=proposal_limit, output=out)
    decoders = {}; decoder_ids = {'E3': [], 'E4': []}
    for arm in ('E3', 'E4'):
        for lr in rates:
            key = candidate_id({'arm': arm, 'lr': lr, 'seed': training_seed, 'epochs': epochs, 'patience': patience})
            net, audit = fit_decoder(arr['initial_features'], arr['targets_'+arm], arr['group_ids'], arr['roles'],
                lengths=arr['lengths_'+arm], arm=arm, critic_parameters=critic_count, seed=training_seed,
                epochs=epochs, patience=patience, lr=lr, device=device)
            save_decoder(net, audit, out/f'decoders/{key}.pt'); write_json(out/f'decoders/{key}.json', audit)
            decoders[key] = net; decoder_ids[arm].append(key)
    registry = variants(config, teacher, family, decoder_ids)
    write_json(out/'variant_registry.json', registry)
    bound_spec = _read(costs_path) if costs_path else None
    if bound_spec is not None:
        if bound_spec.get('work_unit') != 'multiply_add_products' or bound_spec.get('fitted_role') != 'Dexplore':
            raise ValueError('Operation bounds must be declared on Dexplore in the instrumented product unit')
        write_json(out/'cost_bounds.json', bound_spec)
    backend = StudyBackend(root, data, config, teacher, comparator, exp['ensemble_members'], device,
                           decoders, costs=None if bound_spec is None else bound_spec['costs'])
    frozen = {s: sha256_state_dict(m) for s, m in backend.models.items()}
    decisions = select_development(backend, data, partition, registry, contract, out)
    selection_hash = sha256(out/'development_selection.json')
    write_json(out/'selection_lock.json', {'development_selection_sha256': selection_hash,
        'variant_registry_sha256': sha256(out/'variant_registry.json'),
        'locked_before_comparison_label_access': True, 'decisions': decisions})
    selected = {a: next((r for r in registry[a] if r['configuration_id'] == decisions[a]['selected']), None)
                for a in registry}
    evaluate_selected(backend, data, config, exp, contract, selected, out)
    if sha256(out/'development_selection.json') != selection_hash:
        raise RuntimeError('Development decision changed during independent comparison')
    for s, model in backend.models.items():
        if sha256_state_dict(model) != frozen[s]:
            raise RuntimeError('Frozen predictor changed')
    verify_stage_chain(root, 'confirm', expected_source_sha256=current_source_sha256())
    seal(out)
    return out


def evaluate_selected(backend, data, config, exp, contract, selected, out):
    names = ('qC', 'feedback', 'E3', 'E4', 'E5', 'E6')
    parts = {a: [] for a in names}; ids_all = []; units_all = []; y_all = []; input_ids = []
    ledger = []; timings = []; per_input = []
    for eid, episode in episodes(data, 'compare', exp['max_graphs']):
        ids = np.asarray(episode.splits['compare'], dtype=np.int64)
        if not len(ids):
            continue
        qc = backend.execute(episode, {'kind': 'policy', 'candidate': backend.comparator})
        executed = {'qC': qc, 'feedback': backend.execute(episode, {'kind': 'policy', 'candidate': backend.teacher})}
        for arm, spec in selected.items():
            if spec is None:
                executed[arm] = {'prediction': None, 'traces': [], 'status': 'NO_FEASIBLE_DEVELOPMENT_CONFIGURATION',
                                  'deployed_work': 0., 'executed_work': 0., 'support_qualified': False,
                                  'wall_seconds': 0., 'embedded_qualification_work': 0., 'separate_qualification_work': 0.}
            else:
                executed[arm] = backend.execute(episode, spec, qc_reference_work=qc['deployed_work'],
                                                hard_cap_ratio=contract.hard_cap_ratio)
        # The predictions and frozen decisions exist before any independent comparison labels are read.
        for arm, result in executed.items():
            p = result['prediction']; parts[arm].append(None if p is None else p[ids])
            trace = {'episode_id': eid, 'arm': arm, 'configuration_id': None if arm not in selected or selected[arm] is None
                     else selected[arm]['configuration_id'], 'status': result['status'],
                     'feasible': result['support_qualified'], 'member_traces': result['traces'],
                     'multiply_add_work': result['executed_work'],
                     'deployed_multiply_add_work': result['deployed_work'],
                     'actual_state_support_audit': {'feasible': result['support_qualified'],
                         'qualification_multiply_add_work': result['separate_qualification_work']}}
            ledger.append(trace)
            timings.append({'episode_id': eid, 'member_seed': None, 'arm': arm,
                            'multiply_add_work': result['executed_work'], 'deployed_multiply_add_work': result['deployed_work'],
                            'wall_seconds': result['wall_seconds']})
            per_input.append({'episode_id': eid, 'arm': arm, 'status': result['status'],
                              'deployed_work': result['deployed_work'], 'qc_deployed_work': qc['deployed_work'],
                              'expected_ceiling_ratio': contract.expected_ratio,
                              's3_hard_cap': contract.hard_cap_ratio * qc['deployed_work'],
                              'support_qualified': result['support_qualified']})
        global_ids = episode.global_sample_ids
        ids_all.extend(ids.tolist() if global_ids is None else global_ids[ids].tolist())
        units_all.extend(episode.cluster_ids[ids].tolist()); input_ids.extend([eid]*len(ids))
        y_all.append(subset_labels(data, episode, 'compare', ids))
    if not y_all:
        raise ValueError('No reserved comparison observations')
    y = np.concatenate(y_all); units = np.asarray(units_all); ids = np.asarray(ids_all)
    observed = {'observation_id': ids, 'unit_id': units, 'input_id': np.asarray(input_ids), 'y_true': y}
    summary = {}; unit_scores = {}
    for arm, values in parts.items():
        if any(p is None for p in values):
            summary[arm] = {'status': 'NOT_SCORED', 'scored': False}
            continue
        pred = np.concatenate(values); observed[arm] = pred
        scores = score_observations(pred, y, units, config)
        summary[arm] = {'status': 'EXECUTED', 'scored': True, 'loss': scores['node_ce'],
                        'metric': scores['metric'], 'metric_name': scores['metric_name'],
                        'statistical_units': len(scores['core_ids']),
                        'support_qualified': all(t['feasible'] for t in ledger if t['arm'] == arm),
                        'selected_configuration': None if arm not in selected else selected[arm],
                        'interpretation': 'comparator retained' if arm in selected and selected[arm] and
                        selected[arm].get('origin') == 'locked_comparator' else 'executed selected policy'}
        unit_scores[arm] = scores['loss']; unit_scores['unit_id'] = scores['core_ids']; unit_scores['unit_weight'] = scores['counts']
    for arm in ('E3', 'E4', 'E5', 'E6'):
        if arm in unit_scores:
            unit_scores[arm+'_minus_feedback'] = unit_scores[arm] - unit_scores['feedback']
            unit_scores[arm+'_minus_qC'] = unit_scores[arm] - unit_scores['qC']
    require_arm_contract(exp, data.splits, observations=observed['observation_id'])
    np.savez_compressed(out/'predictions.npz', **observed); np.savez_compressed(out/'paired_unit_losses.npz', **unit_scores)
    write_json(out/'execution_traces.json', jsonable(ledger)); write_json(out/'timings.json', timings)
    write_json(out/'per_input_work.json', per_input)
    work = summarize_work(timings, ledger, ratio=contract.expected_ratio,
                          scored={a: s['scored'] for a, s in summary.items()}, basis='deployment')
    write_json(out/'work_comparisons.json', work)
    for arm in summary:
        summary[arm]['within_declared_mean_work_ceiling'] = work['summary'][arm]['within_mean_ceiling']
    write_json(out/'results.json', {'results': summary, 'evaluation_role': 'D_compare',
        'ensemble_members': exp['ensemble_members'], 'development_selection_sha256': sha256(out/'development_selection.json'),
        'expected_budget': contract.record(), 'official_campaign_complete': False,
        'comparison_failure_policy': 'retain measured outcomes and report failures; never reselect after seeing D_compare',
        'statistical_inference': 'one fixed-ensemble study; campaign-level seed/unit analysis is separate',
        'elapsed_time_scope': 'instrumented calls; no benchmark-latency claim without warm-up/repeats'})
