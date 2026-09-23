"""Matched predictor histories and one-update interventions on fixed model states."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'predictor_dynamic/adaptive-validation/scripts'))
from single_layer import (Captures, Collator, CompletionDataset, HiddenPredictor,
    braw, collect, diagnose, fingerprints, flat, infer, load_predictor, measures,
    pack, pred, setup, weights)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def state(p):
    return {k: v.detach().clone() for k, v in p.state_dict().items()}


def clone_predictor(config, value):
    p = HiddenPredictor(**config).cuda()
    p.load_state_dict(value)
    p.eval()
    return p


def save_predictor(path, config, p):
    torch.save(dict(config=config,
        state={k: v.detach().cpu() for k, v in p.state_dict().items()}), path)


def restore_model(mod, values):
    with torch.no_grad():
        for destination, source in zip(weights(mod)[:2], values):
            destination.copy_(source)


def load_adapter(mod, path):
    values = load_file(str(Path(path) / 'adapter_model.safetensors'))
    a = next(v for k, v in values.items() if k.endswith('lora_A.weight'))
    b = next(v for k, v in values.items() if k.endswith('lora_B.weight'))
    restore_model(mod, (a.cuda(), b.cuda()))


def fit(p, training, validation, mod):
    """Only these two datasets enter optimization/selection; never probe targets."""
    started = now()
    A, B, scale = weights(mod)
    a, b = A.detach(), B.detach()
    protected = [A.detach().clone(), B.detach().clone()]
    d, v = pack(training), pack(validation)
    with torch.no_grad():
        targets = braw(d['x'], d['g'], a, b, scale)
    p.eval()
    initial = diagnose(p, v, a, b, scale)
    best_score, best_epoch, best = initial['selection_score'], 0, state(p)
    curve = [dict(epoch=0, selection_score=best_score)]
    p.requires_grad_(True)
    optimizer = torch.optim.AdamW(p.parameters(), lr=1e-4, weight_decay=.01)
    for epoch in range(1, 31):
        p.train()
        predicted = pred(p, d)
        factors = braw(d['x'], predicted, a, b, scale)
        factor_loss = sum(((x-y).square().sum((1, 2)) /
            y.square().sum((1, 2)).clamp_min(1e-12)).mean()
            for x, y in zip(factors, targets)) / 2
        activation_loss = ((predicted-d['g']).square().sum() /
            d['valid'].sum() / predicted.shape[-1] / p.target_scale.square())
        loss = factor_loss + .25 * activation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(w.grad is None for w in (A, B))
        torch.nn.utils.clip_grad_norm_(p.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        p.eval()
        score = diagnose(p, v, a, b, scale)['selection_score']
        curve.append(dict(epoch=epoch, selection_score=score, training_loss=float(loss)))
        if score < best_score:
            best_score, best_epoch, best = score, epoch, state(p)
    p.load_state_dict(best)
    p.requires_grad_(False)
    p.eval()
    assert all(torch.equal(w, old) for w, old in zip((A, B), protected))
    return dict(selected_epoch=best_epoch, selected_score=best_score,
        curve=curve, seconds=now()-started, optimizer_steps=30,
        selection_evaluations=31, calibration_examples=4, selection_examples=16)


@torch.no_grad()
def predicted_gradient(p, samples, mod):
    d = pack(samples)
    A, B, scale = weights(mod)
    factors = braw(d['x'], pred(p, d), A, B, scale)
    n = int(d['counts'].sum())
    return [g.sum(0) / n for g in factors]


@torch.no_grad()
def gradient_metrics(p, samples, true_sum, mod):
    started = now()
    d = pack(samples)
    A, B, scale = weights(mod)
    metric = diagnose(p, d, A, B, scale)
    predicted = predicted_gradient(p, samples, mod)
    n = sum(s['n'] for s in samples)
    metric['batch_vs_autograd'] = measures(flat(predicted), flat([v/n for v in true_sum]))
    metric['examples'] = len(samples)
    metric['seconds'] = now()-started
    return metric


def reconstruction_audit(samples, true_sum, mod):
    with torch.no_grad():
        d = pack(samples)
        A, B, scale = weights(mod)
        raw = braw(d['x'], d['g'], A, B, scale)
        result = measures(flat([v.sum(0) for v in raw]), flat(true_sum))
    assert result['relative_l2'] < .05 and result['cosine'] > .995, result
    return result


def loss_items(tok, max_length):
    ds = CompletionDataset(HERE / 'data/loss256.jsonl', tok, max_length)
    rows = [json.loads(s) for s in (HERE / 'data/loss256.jsonl').read_text().splitlines()]
    masks = []
    for row, item in zip(rows, ds.items):
        answer = row['solution']
        left = answer.rfind('\\boxed{') + 7
        right = answer.rfind('}')
        assert 7 <= left < right
        encoded = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
        supervised = [v for v in item['labels'] if v != -100]
        assert encoded['input_ids'] == supervised[:-1]
        prefix = len(item['labels']) - len(supervised)
        mask = [False] * len(item['labels'])
        for i, (start, end) in enumerate(encoded['offset_mapping']):
            mask[prefix+i] = start < right and end > left
        assert any(mask)
        masks.append(mask)
    return ds, masks, [row['source_index'] for row in rows]


@torch.inference_mode()
def evaluate_loss(model, dataset, expression_masks, indices, collate):
    started = now()
    model.eval()
    records = []
    for start in range(0, len(dataset), 8):
        selected = list(range(start, min(start+8, len(dataset))))
        batch = {k: v.cuda() for k, v in collate([dataset[i] for i in selected]).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           use_cache=False)
        labels = batch['labels'][:, 1:]
        active = labels != -100
        token_loss = torch.nn.functional.cross_entropy(
            output.logits[:, :-1][active].float(), labels[active], reduction='none')
        full = torch.zeros_like(labels, dtype=torch.float32)
        full[active] = token_loss
        for local, index in enumerate(selected):
            expression = torch.zeros_like(active[local])
            row_mask = expression_masks[index][1:]
            expression[:len(row_mask)] = torch.tensor(row_mask, device='cuda')
            assert not (expression & ~active[local]).any()
            records.append(dict(source_index=indices[index],
                token_loss_sum=float(full[local].sum()), tokens=int(active[local].sum()),
                expression_loss_sum=float(full[local][expression].sum()),
                expression_tokens=int(expression.sum())))
    tokens = sum(r['tokens'] for r in records)
    expression_tokens = sum(r['expression_tokens'] for r in records)
    return dict(token_mean_ce=sum(r['token_loss_sum'] for r in records)/tokens,
        expression_token_ce=sum(r['expression_loss_sum'] for r in records)/expression_tokens,
        tokens=tokens, expression_tokens=expression_tokens, examples=len(records),
        seconds=now()-started, rows=records)


def apply_update(model, mod, original, gradient):
    restore_model(mod, original)
    params = list(weights(mod)[:2])
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0)
    model.zero_grad(set_to_none=True)
    for param, value in zip(params, gradient):
        param.grad = value.detach().clone()
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True))
    opt.step()
    model.zero_grad(set_to_none=True)
    delta = flat([p.detach()-old for p, old in zip(params, original)])
    return norm, delta.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=(101, 102, 103), required=True)
    args = ap.parse_args()
    manifest = json.loads((HERE / 'data/manifest.json').read_text())
    assert sha(HERE / 'PLAN.md') == manifest['plan_sha256']
    for name, split in manifest['splits'].items():
        assert sha(HERE / 'data' / f'{name}.jsonl') == split['sha256']
    for path, digest in manifest['dependencies'].items():
        assert sha(path) == digest, path
    starts = manifest['starts'][str(args.seed)]
    assert sha(starts['predictor']) == starts['predictor_sha256']
    out = HERE / 'models' / f'u{args.seed}'
    out.mkdir(exist_ok=False)
    result_path = HERE / 'results' / f'u{args.seed}.json'
    model, tok, mods = setup(adapter=starts['model_states']['0']['path'], rng=args.seed)
    mod = mods[8]
    capture = Captures(mods)
    original_base = fingerprints(model)
    initial_p, info = load_predictor(starts['predictor'])
    initial_state = state(initial_p)
    predictor_config = info['config']
    carry = clone_predictor(predictor_config, initial_state)
    initial_p.requires_grad_(False)
    ds_loss, expression_masks, loss_indices = loss_items(tok, model.config.max_position_embeddings)
    collate = Collator(tok.pad_token_id)
    result = dict(seed=args.seed, predictor_seed=args.seed+22,
        plan_sha256=manifest['plan_sha256'], manifest_sha256=sha(HERE / 'data/manifest.json'),
        script_sha256=sha(__file__), history=[], probes=[], baselines={}, complete=False,
        protocol=dict(input_truncation=False, primary_model_step=32, probes=4,
            calibration_lr=1e-4, calibration_steps=30, lora_lr=3e-4,
            optimizer_reset_per_episode=True, matched_main_model_states=True))
    started = now()

    def gather(split):
        t = now()
        ds = CompletionDataset(HERE / 'data' / f'{split}.jsonl', tok,
                               model.config.max_position_embeddings)
        samples, true = collect(model, tok, mods, capture, ds, list(range(len(ds))))
        assert all(p.grad is None for p in model.parameters())
        return samples[8], true[8], now()-t

    def losses():
        return evaluate_loss(model, ds_loss, expression_masks, loss_indices, collate)

    for step in (0, 1, 2, 4, 8, 16, 32):
        adapter = starts['model_states'][str(step)]
        assert sha(Path(adapter['path']) / 'adapter_model.safetensors') == adapter['sha256']
        load_adapter(mod, adapter['path'])
        model.zero_grad(set_to_none=True)
        model.eval()
        original = [v.detach().clone() for v in weights(mod)[:2]]
        if step in (16, 32):
            baseline = losses()
            result['baselines'][str(step)] = baseline
            for probe in range(4):
                prefix = f'probe_s{step}_p{probe}'
                calibration, cal_true, cal_seconds = gather(prefix + '_cal')
                validation, _, val_seconds = gather(prefix + '_val')
                # Diagnostic targets are obtained once, held out of fit()/selection.
                held, held_true, held_seconds = gather(prefix + '_held')
                samples = calibration + held
                true_sum = [x+y for x, y in zip(cal_true, held_true)]
                count = sum(s['n'] for s in samples)
                true_gradient = [v/count for v in true_sum]
                row = dict(model_step=step, probe=probe,
                    starting_adapter_sha256=adapter['sha256'],
                    collection_seconds=dict(calibration=cal_seconds, selection=val_seconds,
                                            diagnostic_heldout=held_seconds),
                    reconstruction_audit=reconstruction_audit(samples, true_sum, mod),
                    methods={})
                gradients = {}
                for method, value in [('carry', state(carry)), ('reset', initial_state)]:
                    p = clone_predictor(predictor_config, value)
                    before = gradient_metrics(p, held, held_true, mod)
                    fit_record = fit(p, calibration, validation, mod)
                    after = gradient_metrics(p, held, held_true, mod)
                    full = gradient_metrics(p, samples, true_sum, mod)
                    tick = now()
                    gradients[method] = predicted_gradient(p, samples, mod)
                    prediction_seconds = now()-tick
                    row['methods'][method] = dict(before_heldout=before, fit=fit_record,
                        heldout=after, full_batch=full, prediction_seconds=prediction_seconds)
                    if probe == 0:
                        save_predictor(out / f'predictor_s{step}_p{probe}_{method}.pt', predictor_config, p)
                    del p
                gradients['frozen'] = predicted_gradient(initial_p, samples, mod)
                row['methods']['frozen'] = dict(
                    heldout=gradient_metrics(initial_p, held, held_true, mod),
                    full_batch=gradient_metrics(initial_p, samples, true_sum, mod))
                gradients['oracle'] = true_gradient
                row['methods']['oracle'] = dict()
                deltas = {}
                for method in ('carry', 'reset', 'frozen', 'oracle'):
                    assert all(p.grad is None for p in model.parameters())
                    norm, delta = apply_update(model, mod, original, gradients[method])
                    deltas[method] = delta
                    intervention = row['methods'][method]
                    intervention['raw_gradient_norm'] = norm
                    intervention['after_update'] = losses()
                    intervention['ce_improvement'] = (baseline['token_mean_ce'] -
                        intervention['after_update']['token_mean_ce'])
                    intervention['expression_ce_improvement'] = (baseline['expression_token_ce'] -
                        intervention['after_update']['expression_token_ce'])
                    if step == 32 and probe == 0:
                        destination = out / f's32_p0_{method}'
                        model.save_pretrained(destination)
                        intervention['adapter_path'] = str(destination)
                        intervention['adapter_sha256'] = sha(destination / 'adapter_model.safetensors')
                    restore_model(mod, original)
                for method in ('carry', 'reset', 'frozen'):
                    row['methods'][method]['update_vs_oracle'] = measures(deltas[method], deltas['oracle'])
                assert all(torch.equal(v, old) for v, old in zip(weights(mod)[:2], original))
                result['probes'].append(row)
                write(result_path, result)
                print(json.dumps(dict(seed=args.seed, step=step, probe=probe,
                    carry_epoch=row['methods']['carry']['fit']['selected_epoch'],
                    reset_epoch=row['methods']['reset']['fit']['selected_epoch'],
                    ce_gain={k: v['ce_improvement'] for k, v in row['methods'].items()},
                    elapsed=now()-started)), flush=True)

        if step != 32:
            calibration, _, cal_seconds = gather(f'history_s{step}_cal')
            validation, _, val_seconds = gather(f'history_s{step}_val')
            history = dict(model_step=step, collection_seconds=dict(
                calibration=cal_seconds, selection=val_seconds))
            history['carry'] = fit(carry, calibration, validation, mod)
            reset = clone_predictor(predictor_config, initial_state)
            history['reset'] = fit(reset, calibration, validation, mod)
            if step == 0:
                assert all(torch.equal(v, reset.state_dict()[k]) for k, v in carry.state_dict().items())
            history['predictor_equal_at_first_episode'] = step == 0
            save_predictor(out / f'history_s{step}_carry.pt', predictor_config, carry)
            save_predictor(out / f'history_s{step}_reset.pt', predictor_config, reset)
            result['history'].append(history)
            del reset
            write(result_path, result)
            print(json.dumps(dict(seed=args.seed, history_step=step,
                carry_epoch=history['carry']['selected_epoch'],
                reset_epoch=history['reset']['selected_epoch'], elapsed=now()-started)), flush=True)

    assert fingerprints(model) == original_base
    capture.close()
    result.update(complete=True, frozen_base_unchanged=True, elapsed_seconds=now()-started,
        budgets=dict(history_episodes=6, probe_episodes=8,
            history_calibration_labels=24, history_selection_labels=96,
            per_probe_calibration_labels=4, per_probe_selection_labels=16,
            diagnostic_labels_per_probe=28,
            predictor_optimizer_steps_per_method_per_episode=30,
            diagnostic_targets_used_for_selection=False))
    write(result_path, result)
    write(out / 'audit.json', dict(complete=True, frozen_base_unchanged=True,
        input_truncation=False, probes=len(result['probes']), history=len(result['history']),
        plan_sha256=manifest['plan_sha256']))
    print(json.dumps(dict(seed=args.seed, complete=True, elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
