"""Post-confirmation normalized residual experiment; no file readers or official score.

Each fit accepts only declared training contexts. A one-context inner fit cannot
estimate context adaptation and explicitly uses zero learned residual.
"""
from __future__ import annotations
import numpy as np

METHODS = ('pooled', 'neighbor', 'low_rank', 'neural')
CONTEXTS = ('K562', 'RPE1', 'HepG2')


def counts(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError('Finite nonnegative integral cell counts required')
    return x


def profile(x, total=10000.):
    x = counts(x); depth = x.sum(1)
    if (depth <= 0).any():
        raise ValueError('Zero-depth cells cannot define a normalized expression profile')
    normalized = x / depth[:, None] * total
    return np.log1p(normalized.mean(0)), float(depth.mean())


def mean_error(prediction, truth):
    p = np.asarray(prediction, dtype=float); y = np.asarray(truth, dtype=float)
    if p.shape != y.shape or p.ndim != 2 or not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError('Aligned finite nonnegative group means required')
    return np.mean((np.log1p(p) - np.log1p(y)) ** 2, axis=1)


def _projection(x, rank):
    # Rank/axes depend only on the supplied training matrix.
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    n = min(rank, int(np.sum(singular > 1e-10)))
    return vt[:n].T


def _network_fit(x, y, options):
    rng = np.random.default_rng(options['seed']); width = options['neural_hidden']
    params = [rng.normal(0, .05, (x.shape[1], width)), np.zeros(width),
              np.zeros((width, y.shape[1])), np.zeros(y.shape[1])]
    first = [np.zeros_like(v) for v in params]; second = [np.zeros_like(v) for v in params]
    curve = []; lr = options['neural_learning_rate']; wd = options['neural_weight_decay']
    for step in range(1, options['neural_steps'] + 1):
        h = np.tanh(x @ params[0] + params[1]); output = h @ params[2] + params[3]
        error = output-y; g = 2*error/error.size
        gh = (g @ params[2].T)*(1-h*h)
        grads = [x.T @ gh, gh.sum(0), h.T @ g, g.sum(0)]
        for i in range(4):
            grad = grads[i] + wd*params[i]
            first[i] = .9*first[i]+.1*grad; second[i] = .999*second[i]+.001*grad*grad
            params[i] -= lr*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 20 == 0:
            loss = float(np.mean(error*error))
            if not np.isfinite(loss): raise ValueError('Nonfinite residual optimization')
            curve.append({'step': step, 'training_loss': loss})
    return params, curve


def fit(training, targets, options):
    contexts = sorted(training)
    if not contexts or len(contexts) > 2 or any(c not in CONTEXTS for c in contexts):
        raise ValueError('One or two declared development training contexts required')
    if len(set(targets)) != len(targets) or not targets: raise ValueError('Distinct targets required')
    z = []; delta = []
    for context in contexts:
        groups = training[context]
        if set(groups) != {'control', *targets}: raise ValueError('Exact training target groups required')
        control, _ = profile(groups['control'], options['target_sum']); z.append(control)
        delta.append(np.stack([profile(groups[t], options['target_sum'])[0]-control for t in targets]))
    z = np.stack(z); delta = np.stack(delta)
    result = {'contexts': contexts, 'targets': list(targets), 'control_profiles': z, 'deltas': delta,
              'options': dict(options), 'zero_residual_fallback': len(contexts) < 2, 'curve': []}
    if len(contexts) < 2: return result
    center = z.mean(0); context_basis = _projection(z-center, 1)
    target_mean = delta.mean(0); target_basis = _projection(target_mean, options['rank'])
    target_features = target_mean @ target_basis
    cfeatures = (z-center) @ context_basis
    cscale = np.maximum(np.std(cfeatures, axis=0), 1e-6); cfeatures /= cscale
    tscale = np.maximum(np.std(target_features, axis=0), 1e-6); target_features /= tscale
    def features(c):
        repeated = np.broadcast_to(c, (len(targets), len(c)))
        return np.column_stack([np.ones(len(targets)), repeated, target_features,
                                (repeated[:, :, None]*target_features[:, None, :]).reshape(len(targets), -1)])
    x = np.concatenate([features(c) for c in cfeatures])
    # Each training label uses the opposite source context; outer-held data absent.
    residual = np.concatenate([delta[i]-options['shrinkage']*delta[1-i] for i in range(2)])
    response_basis = _projection(residual, options['rank'])
    if response_basis.shape[1] == 0:
        result.update(zero_residual_fallback=True, fallback_reason='zero_rank_training_effect')
        return result
    y = residual @ response_basis
    ridge = np.linalg.solve(x.T@x+options['ridge_alpha']*np.eye(x.shape[1]), x.T@y)
    params, curve = _network_fit(x, y, options) if options.get('fit_neural', True) else (None, [])
    result.update(center=center, context_basis=context_basis, context_scale=cscale,
                  target_features=target_features, response_basis=response_basis, ridge=ridge,
                  neural=params, curve=curve)
    return result


def predict(fitted, held_controls, targets, method):
    if method not in METHODS: raise ValueError('Unknown residual family')
    options = fitted['options']; q, depth = profile(held_controls, options['target_sum'])
    z = fitted['control_profiles']; delta = fitted['deltas']
    distances = np.mean((z-q)**2, axis=1)
    weights = np.ones(len(z))/len(z)
    if method == 'neighbor' and len(z)>1:
        scale = max(float(np.mean((z[0]-z[1])**2)), 1e-8)
        logits = -distances/(2*scale); weights = np.exp(logits-logits.max()); weights /= weights.sum()
    transferred = options['shrinkage']*np.tensordot(weights, delta, axes=(0, 0))
    residual = np.zeros_like(transferred)
    if method in ('low_rank', 'neural') and not fitted['zero_residual_fallback']:
        c = (q-fitted['center'])@fitted['context_basis']/fitted['context_scale']
        c = np.clip(c, -options['context_clip'], options['context_clip'])
        t = fitted['target_features']; repeated = np.broadcast_to(c, (len(t), len(c)))
        x = np.column_stack([np.ones(len(t)), repeated, t,
                             (repeated[:, :, None]*t[:, None, :]).reshape(len(t), -1)])
        if method == 'low_rank': latent = x @ fitted['ridge']
        else:
            if fitted['neural'] is None: raise ValueError('Neural residual was not fitted')
            a, b, w, bias = fitted['neural']; latent = np.tanh(x@a+b)@w+bias
        residual = options['residual_strength']*(latent @ fitted['response_basis'].T)
    pred = []; features = []; fallback = []
    for target in targets:
        if target not in fitted['targets'] or target == 'control':
            effect = np.zeros_like(q); variance = 0.; fallback.append(target)
        else:
            i = fitted['targets'].index(target); effect = transferred[i]+residual[i]
            variance = float(np.mean(np.var(delta[:, i], axis=0)))
        normalized = np.expm1(np.clip(q+effect, 0., 20.))
        normalized *= options['target_sum']/max(float(normalized.sum()), 1e-12)
        pred.append(normalized)
        features.append([variance, float(np.mean(effect**2)), float(np.mean(distances))])
    return {'normalized_means': np.stack(pred), 'raw_depth_matched_means': np.stack(pred)*depth/options['target_sum'],
            'features': np.asarray(features), 'fallback_targets': fallback,
            'training_contexts': fitted['contexts'], 'zero_residual_fallback': fitted['zero_residual_fallback']}


def calibrate(training, targets, options, method):
    """Leave-one-training-context-out features; outer held context never accepted."""
    if len(training) != 2: raise ValueError('Calibration requires exactly two outer-training contexts')
    features = []; errors = []; lineage = []
    for held in sorted(training):
        permitted = {c: v for c, v in training.items() if c != held}
        model = fit(permitted, targets, options)
        prediction = predict(model, training[held]['control'], targets, method)
        truth = np.stack([np.expm1(profile(training[held][t], options['target_sum'])[0]) for t in targets])
        features.extend(prediction['features']); errors.extend(mean_error(prediction['normalized_means'], truth))
        lineage.append({'inner_held': held, 'base_training_contexts': model['contexts'],
                        'zero_context_residual': model['zero_residual_fallback']})
    x = np.log1p(np.asarray(features)); center = x.mean(0); scale = np.maximum(x.std(0), 1e-8)
    x = (x-center)/scale; y = np.log(np.asarray(errors)+1e-8); intercept = float(y.mean())
    coef = np.linalg.solve(x.T@x+np.eye(x.shape[1]), x.T@(y-intercept))
    return {'center': center, 'scale': scale, 'coef': coef, 'intercept': intercept,
            'training_contexts': sorted(training), 'lineage': lineage,
            'inner_errors': list(map(float, errors))}


def calibrated_error(features, calibration):
    x = (np.log1p(features)-calibration['center'])/calibration['scale']
    return np.maximum(0, np.exp(np.clip(calibration['intercept']+x@calibration['coef'], -40, 20))-1e-8)
