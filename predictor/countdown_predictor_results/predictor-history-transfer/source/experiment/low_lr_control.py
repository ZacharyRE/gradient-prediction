"""Outcome-informed, loss-only one-step LR diagnostic; preserve primary artifacts."""
import argparse
import json
from pathlib import Path

import torch

from run_seed import (HERE, Captures, Collator, CompletionDataset, collect,
    clone_predictor, evaluate_loss, fingerprints, fit, load_adapter, load_predictor,
    loss_items, now, predicted_gradient, restore_model, setup, sha, state, weights, write)


def main():
    manifest = json.loads((HERE / 'data/manifest.json').read_text())
    folder = HERE / 'results/low_lr'
    folder.mkdir(exist_ok=True)
    for seed in (101, 102, 103):
        output = folder / f'u{seed}.json'
        assert not output.exists()
        primary = json.loads((HERE / 'results' / f'u{seed}.json').read_text())
        assert primary['complete']
        starts = manifest['starts'][str(seed)]
        model, tok, mods = setup(adapter=starts['model_states']['16']['path'], rng=seed)
        mod = mods[8]
        base_before = fingerprints(model)
        capture = Captures(mods)
        initial, info = load_predictor(starts['predictor'])
        initial.requires_grad_(False)
        ds_loss, masks, ids = loss_items(tok, model.config.max_position_embeddings)
        collate = Collator(tok.pad_token_id)
        result = dict(seed=seed, lr=1e-4, exploratory=True, loss_set_reused=True,
            control_protocol_sha256=sha(HERE / 'EXPLORATORY_CONTROL.md'),
            primary_result_sha256=sha(HERE / 'results' / f'u{seed}.json'),
            complete=False, probes=[], baselines={})
        started = now()

        def gather(name):
            ds = CompletionDataset(HERE / 'data' / f'{name}.jsonl', tok,
                                   model.config.max_position_embeddings)
            samples, gradients = collect(model, tok, mods, capture, ds, list(range(len(ds))))
            return samples[8], gradients[8]

        def losses():
            return evaluate_loss(model, ds_loss, masks, ids, collate)

        for step, history_step in [(16, 8), (32, 16)]:
            load_adapter(mod, starts['model_states'][str(step)]['path'])
            original = [v.detach().clone() for v in weights(mod)[:2]]
            baseline = losses()
            assert abs(baseline['token_mean_ce'] - primary['baselines'][str(step)]['token_mean_ce']) < 1e-7
            result['baselines'][str(step)] = baseline
            history_p, _ = load_predictor(HERE / 'models' / f'u{seed}/history_s{history_step}_carry.pt')
            for probe in range(4):
                old = next(r for r in primary['probes'] if r['model_step'] == step and r['probe'] == probe)
                prefix = f'probe_s{step}_p{probe}'
                calibration, cal_true = gather(prefix + '_cal')
                validation, _ = gather(prefix + '_val')
                held, held_true = gather(prefix + '_held')
                samples = calibration + held
                n = sum(s['n'] for s in samples)
                gradients = dict(oracle=[(a+b)/n for a, b in zip(cal_true, held_true)])
                row = dict(model_step=step, probe=probe, methods={}, calibration_reproduced=True)
                for method, source in [('carry', history_p), ('reset', initial)]:
                    p = clone_predictor(info['config'], state(source))
                    fit_record = fit(p, calibration, validation, mod)
                    reference = old['methods'][method]['fit']
                    assert fit_record['selected_epoch'] == reference['selected_epoch']
                    assert abs(fit_record['selected_score'] - reference['selected_score']) < 1e-6
                    if probe == 0:
                        saved = torch.load(HERE / 'models' / f'u{seed}/predictor_s{step}_p0_{method}.pt',
                                           map_location='cuda', weights_only=False)['state']
                        assert all(torch.equal(v, saved[k]) for k, v in p.state_dict().items())
                    gradients[method] = predicted_gradient(p, samples, mod)
                    del p
                gradients['frozen'] = predicted_gradient(initial, samples, mod)
                for method in ('carry', 'reset', 'frozen', 'oracle'):
                    restore_model(mod, original)
                    params = list(weights(mod)[:2])
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
                    model.zero_grad(set_to_none=True)
                    for p, gradient in zip(params, gradients[method]):
                        p.grad = gradient.detach().clone()
                    norm = float(torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True))
                    optimizer.step()
                    model.zero_grad(set_to_none=True)
                    after = losses()
                    row['methods'][method] = dict(after_update=after, raw_gradient_norm=norm,
                        ce_improvement=baseline['token_mean_ce']-after['token_mean_ce'],
                        expression_ce_improvement=baseline['expression_token_ce']-after['expression_token_ce'])
                    restore_model(mod, original)
                result['probes'].append(row)
                write(output, result)
                print(json.dumps(dict(seed=seed, step=step, probe=probe,
                    ce_gain={method: v['ce_improvement'] for method, v in row['methods'].items()},
                    elapsed=now()-started)), flush=True)
            del history_p
        assert fingerprints(model) == base_before
        capture.close()
        result.update(complete=True, frozen_base_unchanged=True, elapsed_seconds=now()-started)
        write(output, result)
        print(json.dumps(dict(seed=seed, complete=True, seconds=result['elapsed_seconds'])), flush=True)
        del model, mod, mods, capture, initial, gradients, params, optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
