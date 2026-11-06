"""E3 initial-input schedule prediction and E4 initial-input distillation.

A complete action sequence is decoded before any controlled plant transition.
Both learners use the same autoregressive architecture and development-only
partition. E3 targets are selected by an explicit development oracle; E4 targets
are produced by the frozen teacher. Dynamic last-safe A9 targets are converted
to source-reset A8, with the count recorded, matching the public open-loop action
contract. This conversion is not claimed to preserve teacher predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderSpec:
    input_dim: int
    context_width: int
    hidden_dim: int
    max_depth: int
    actions: int = 9  # 9: A0--A8. 10: same actions plus internal STOP token 9 (not A9).

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value < 1:
                raise ValueError('Decoder dimensions must be positive integers')
        if self.actions not in (9, 10):
            raise ValueError('Vocabulary is A0--A8 with optional internal STOP token')


class ScheduleDecoder(nn.Module):
    """Context-MLP + GRUCell decoder, using only H0/graph features and prior IDs."""
    def __init__(self, spec: DecoderSpec):
        super().__init__()
        self.spec = spec
        self.context = nn.Sequential(nn.Linear(spec.input_dim, spec.context_width),
                                     nn.SiLU(), nn.Linear(spec.context_width, spec.hidden_dim), nn.Tanh())
        self.recurrent = nn.GRUCell(spec.actions + 2, spec.hidden_dim)
        self.output = nn.Linear(spec.hidden_dim, spec.actions)
        self.register_buffer('feature_mean', torch.zeros(spec.input_dim))
        self.register_buffer('feature_scale', torch.ones(spec.input_dim))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def context_products(self):
        s = self.spec
        return s.input_dim * s.context_width + s.context_width * s.hidden_dim

    @property
    def step_products(self):
        s = self.spec
        return 3 * s.hidden_dim * (s.actions + 2 + s.hidden_dim) + s.hidden_dim * s.actions

    def start(self, features):
        if features.ndim != 2 or features.shape[1] != self.spec.input_dim or not torch.isfinite(features).all():
            raise ValueError('Initial-input features must be a finite [batch, input_dim] tensor')
        return self.context((features - self.feature_mean) / self.feature_scale)

    def step(self, hidden, previous_action, depth):
        if not 0 <= depth < self.spec.max_depth:
            raise ValueError('Decode depth outside registered horizon')
        if previous_action.shape != (len(hidden),) or torch.any((previous_action < -1) | (previous_action >= self.spec.actions)):
            raise ValueError('Invalid previous action')
        prior = F.one_hot(previous_action.clamp_min(0), self.spec.actions).to(hidden.dtype)
        prior = prior * (previous_action >= 0)[:, None]
        relative = hidden.new_full((len(hidden), 1), depth / self.spec.max_depth)
        inputs = torch.cat([prior, relative, 1 - relative], dim=1)
        hidden = self.recurrent(inputs, hidden)
        return self.output(hidden), hidden

    def teacher_forced(self, features, targets):
        if targets.ndim != 2 or targets.shape != (len(features), self.spec.max_depth):
            raise ValueError('Targets must cover each registered decode depth')
        hidden = self.start(features)
        previous = torch.full((len(features),), -1, dtype=torch.long, device=features.device)
        result = []
        for depth in range(self.spec.max_depth):
            logits, hidden = self.step(hidden, previous, depth)
            result.append(logits)
            previous = targets[:, depth]
        return torch.stack(result, 1)

    @torch.no_grad()
    def decode(self, features, support=None):
        """support(depth, prefix) is optional and must be initial/model-state only.

        Empty masks recover with A8; tie order is the lower action ID. No plant
        state, measurements, or labels are arguments to this API.
        """
        self.eval()
        features = torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device)
        if features.ndim == 1:
            features = features[None]
        if len(features) != 1:
            raise ValueError('Decode one graph episode at a time')
        hidden = self.start(features)
        previous = torch.full((1,), -1, dtype=torch.long, device=features.device)
        actions, masks, recoveries = [], [], []
        stopped = False
        decode_steps = 0
        for depth in range(self.spec.max_depth):
            logits, hidden = self.step(hidden, previous, depth)
            decode_steps += 1
            mask = list(range(9)) if support is None else list(support(depth, tuple(actions)))
            if any(type(a) is not int or not 0 <= a < 9 for a in mask) or len(set(mask)) != len(mask):
                raise ValueError('Open-loop mask contains duplicate or invalid action IDs')
            # STOP is a decoder symbol, not the dynamic last-safe action A9.
            # A positive terminal depth matches the registered stopping family.
            choice_mask = mask + ([9] if self.spec.actions == 10 and depth > 0 else [])
            if not choice_mask:
                action = 8
                recoveries.append(depth)
            else:
                values = logits[0].cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError('Nonfinite initial-input policy score')
                action = min(choice_mask, key=lambda a: (-float(values[a]), a))
            if self.spec.actions == 10 and action == 9:
                stopped = True
                break
            actions.append(action); masks.append(mask)
            previous.fill_(action)
        return actions, {'initial_input_only': True, 'support_masks': masks,
                         'empty_support_recovery_depths': recoveries,
                         'decoder_parameters': self.parameter_count,
                         'decoder_multiply_adds': self.context_products + decode_steps * self.step_products,
                         'decode_steps': decode_steps, 'terminal_depth': len(actions),
                         'stop_symbol_selected': stopped, 'stop_symbol_is_not_A9': True,
                         'intermediate_observations_used': 0, 'plan_fixed_before_execution': True}


def count_parameters(spec):
    s = spec
    return ((s.input_dim + 1) * s.context_width + (s.context_width + 1) * s.hidden_dim
            + 3 * s.hidden_dim * (s.actions + 2 + s.hidden_dim + 2)
            + (s.hidden_dim + 1) * s.actions)


def capacity_match(input_dim, max_depth, critic_parameters, tolerance=.01, *, actions=9):
    """Choose real layer widths; never pad with unused parameters."""
    if type(critic_parameters) is not int or critic_parameters < 1 or not 0 <= tolerance <= .1:
        raise ValueError('Invalid capacity-matching contract')
    if actions not in (9, 10):
        raise ValueError('Invalid schedule vocabulary')
    choices = []
    for hidden in range(4, 193):
        # Count is affine in context width, so only adjacent integer widths matter.
        constant = 3 * hidden * (actions + 2 + hidden + 2) + hidden + (hidden + 1) * actions
        guess = (critic_parameters - constant) / (input_dim + hidden + 1)
        for context in {max(1, min(512, math.floor(guess))), max(1, min(512, math.ceil(guess)))}:
            spec = DecoderSpec(input_dim, context, hidden, max_depth, actions=actions)
            count = count_parameters(spec)
            choices.append((abs(count - critic_parameters), count, hidden, context, spec))
    _, count, _, _, spec = min(choices, key=lambda v: v[:4])
    mismatch = abs(count / critic_parameters - 1)
    if mismatch > tolerance:
        raise ValueError(f'No declared decoder within capacity tolerance: mismatch={mismatch:.5f}')
    return spec, {'critic_parameters': critic_parameters, 'decoder_parameters': count,
                  'relative_parameter_mismatch': mismatch, 'allowed_mismatch': tolerance,
                  'parameter_matching': 'nearest active architecture; no unused parameters'}


def _validate_training(features, targets, groups, roles, arm):
    x = np.asarray(features, dtype=np.float32)
    t = np.asarray(targets)
    g = np.asarray(groups).astype(str)
    role = np.asarray(roles).astype(str)
    if arm not in ('E3', 'E4'):
        raise ValueError('Only E3/E4 fit schedule decoders')
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty initial features required')
    if t.ndim != 2 or len(t) != len(x) or t.dtype.kind not in 'iu' or np.any((t < 0) | (t > 9)):
        raise ValueError('Invalid sequence targets')
    if g.shape != (len(x),) or role.shape != (len(x),) or set(role) != {'Dexplore'}:
        raise ValueError('Schedule fitting accepts Dexplore records only')
    unique = np.unique(g)
    if len(unique) < 2:
        raise ValueError('At least two disjoint fitting groups are needed for internal validation')
    return x, np.where(t == 9, 8, t).astype(np.int64), g, int((t == 9).sum())


def fit_decoder(features, targets, groups, roles, *, arm, critic_parameters, seed=1,
                epochs=100, patience=15, lr=1e-3, batch_size=64, tolerance=.01, device='cpu', lengths=None):
    """Fit on development groups, using disjoint development groups for stopping.

    For a single transductive graph, groups are source checkpoint seeds, not
    fictitious independent graph samples. The caller records this distinction.
    """
    x, t, groups, rewrites = _validate_training(features, targets, groups, roles, arm)
    variable = lengths is not None
    if variable:
        lengths = np.asarray(lengths)
        if lengths.shape != (len(t),) or lengths.dtype.kind not in 'iu' or np.any((lengths < 1) | (lengths > t.shape[1])):
            raise ValueError('One registered positive terminal depth is required per target')
        # The first STOP symbol is scored. Padding after it contributes no loss.
        loss_mask = np.arange(t.shape[1])[None, :] <= lengths[:, None]
        t = t.copy()
        t[np.arange(t.shape[1])[None, :] >= lengths[:, None]] = 9
    else:
        loss_mask = np.ones_like(t, dtype=bool)
    if any(type(v) is not int or v < 1 for v in (epochs, patience, batch_size)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid training configuration')
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    nv = max(1, int(round(.2 * len(unique))))
    valid = np.isin(groups, unique[:nv]); train = ~valid
    spec, capacity = capacity_match(x.shape[1], t.shape[1], critic_parameters, tolerance, actions=10 if variable else 9)
    torch.manual_seed(seed)
    net = ScheduleDecoder(spec).to(device)
    mean = x[train].mean(0); scale = np.maximum(x[train].std(0), 1e-6)
    net.feature_mean.copy_(torch.from_numpy(mean).to(device))
    net.feature_scale.copy_(torch.from_numpy(scale).to(device))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.from_numpy(x); T = torch.from_numpy(t)
    loss_targets = t.copy(); loss_targets[~loss_mask] = -100
    Y = torch.from_numpy(loss_targets)
    train_ids = np.flatnonzero(train); valid_ids = np.flatnonzero(valid)
    best_loss, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(epochs):
        net.train(); total, count = 0., 0
        for batch in np.array_split(rng.permutation(train_ids), max(1, math.ceil(len(train_ids)/batch_size))):
            batch_x, batch_t = X[batch].to(device), T[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net.teacher_forced(batch_x, batch_t)
            loss = F.cross_entropy(logits.flatten(0, 1), Y[batch].to(device).flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite sequence fitting loss')
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            total += float(loss.detach()) * len(batch); count += len(batch)
        net.eval(); val_sum = 0.
        with torch.no_grad():
            for start in range(0, len(valid_ids), batch_size):
                batch = valid_ids[start:start + batch_size]
                logits = net.teacher_forced(X[batch].to(device), T[batch].to(device))
                val_sum += float(F.cross_entropy(logits.flatten(0, 1), Y[batch].to(device).flatten(), ignore_index=-100)) * len(batch)
        val_loss = val_sum / len(valid_ids)
        if not math.isfinite(val_loss):
            raise FloatingPointError('Nonfinite internal validation loss')
        history.append({'epoch': epoch, 'train_sequence_ce': total/count, 'validation_sequence_ce': val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    net.load_state_dict(best_state); net.eval().requires_grad_(False)
    audit = {'arm': arm, 'spec': asdict(spec), 'capacity': capacity, 'seed': seed,
             'training_rows': int(train.sum()), 'validation_rows': int(valid.sum()),
             'training_groups': sorted(map(str,set(groups[train]))), 'validation_groups': sorted(map(str,set(groups[valid]))),
             'fresh_label_role': 'Dexplore', 'feature_normalization_role': 'Dexplore fitting groups only',
             'a9_to_a8_target_conversions': rewrites, 'target_semantics': 'source-reset-normalized complete schedule',
             'target_origin': 'development-oracle selection' if arm == 'E3' else 'frozen feedback teacher',
             'variable_terminal_depths': variable,
             'stop_token': 'internal token 9; distinct from dynamic A9' if variable else None,
             'target_lengths': lengths.tolist() if variable else [t.shape[1]] * len(t),
             'history': history, 'best_validation_sequence_ce': best_loss}
    return net, audit


def save_decoder(net, audit, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema': 'cm-schedule-decoder-1', 'spec': asdict(net.spec),
                'state_dict': {k: v.detach().cpu() for k,v in net.state_dict().items()}, 'audit': audit}, path)


def load_decoder(path, device='cpu'):
    # Only tensors and built-in containers are permitted, unlike legacy model payloads.
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('schema') != 'cm-schedule-decoder-1':
        raise ValueError('Unrecognized schedule checkpoint')
    net = ScheduleDecoder(DecoderSpec(**value['spec'])).to(device)
    net.load_state_dict(value['state_dict']); net.eval().requires_grad_(False)
    return net, value['audit']
