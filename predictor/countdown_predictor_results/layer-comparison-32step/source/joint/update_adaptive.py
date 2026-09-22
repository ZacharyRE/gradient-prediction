from three_layers import *
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['three'], required=True)
    ap.add_argument('--architecture', choices=['independent'], default='independent')
    ap.add_argument('--mode', choices=['adaptive', 'frozen', 'oracle', 'fixed5'], required=True)
    ap.add_argument('--train-seed', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--steps', type=int, required=True)
    a = ap.parse_args()
    cfg = json.loads((EXP / 'configs/experiment.json').read_text())
    out = EXP / 'models' / a.name
    out.mkdir(exist_ok=False)
    m, tok, mods = setup(a.scope, adapter=None, rng=a.seed)
    cap = Captures(mods)
    allparams = [v for mod in mods.values() for v in weights(mod)[:2]]
    base_before = fingerprints(m)
    ds = CompletionDataset(data_path('update'), tok, m.config.max_position_embeddings)
    dev = CompletionDataset(data_path('predictor_dev'), tok, m.config.max_position_embeddings)
    order = list(np.random.RandomState(a.seed).permutation(len(ds)))
    subrng = np.random.RandomState(a.seed + 10000)
    pp = {}
    infos = {}
    if a.mode != 'oracle':
        for l in mods:
            source = EXP / 'models' / f'independent_l{l:02d}_s{a.train_seed}' / 'best.pt'
            pp[l], infos[l] = load_predictor(source)
        for p in pp.values():
            p.requires_grad_(False)
            p.eval()
    initial_states = {l: {k: v.detach().clone() for k, v in p.state_dict().items()} for l, p in pp.items()}
    opt = torch.optim.AdamW(allparams, lr=0.0003, weight_decay=0)
    fixedopts = {l: torch.optim.AdamW(p.parameters(), lr=0.0001, weight_decay=0.01) for l, p in pp.items()} if a.mode == 'fixed5' else {}
    hist = []
    start = time.time()
    write(out / 'config.json', dict(args=vars(a), protocol=cfg, layers=list(mods), validation_ids=list(range(16)), input_truncation=False))

    def fit_layer(layer, train, valid):
        p = pp[layer]
        A, B, s = weights(mods[layer])
        d = pack(train)
        v = pack(valid)
        before = diagnose(p, v, A, B, s)
        bestscore = before['selection_score']
        best = {k: w.detach().clone() for k, w in p.state_dict().items()}
        bestepoch = 0
        scores = [dict(epoch=0, score=bestscore)]
        losses = []
        po = fixedopts[layer] if a.mode == 'fixed5' else torch.optim.AdamW(p.parameters(), lr=0.0001, weight_decay=0.01)
        p.requires_grad_(True)
        with torch.no_grad():
            ta, tb = braw(d['x'], d['g'], A.detach(), B.detach(), s)
        for epoch in range(1, 6 if a.mode == 'fixed5' else 31):
            p.train()
            g = pred(p, d)
            pa, pb = braw(d['x'], g, A.detach(), B.detach(), s)
            loss = sum((((pg - tg).square().sum((1, 2)) / tg.square().sum((1, 2)).clamp_min(1e-12)).mean() for pg, tg in [(pa, ta), (pb, tb)])) / 2
            loss += 0.25 * (g - d['g']).square().sum() / d['valid'].sum() / 896 / p.target_scale.square()
            po.zero_grad(set_to_none=True)
            loss.backward()
            assert all((w.grad is None for w in allparams))
            nn.utils.clip_grad_norm_(p.parameters(), 1.0, error_if_nonfinite=True)
            po.step()
            p.eval()
            losses.append(float(loss.detach()))
            score = diagnose(p, v, A, B, s)['selection_score']
            scores.append(dict(epoch=epoch, score=score))
            if score < bestscore:
                bestscore = score
                bestepoch = epoch
                best = {k: w.detach().clone() for k, w in p.state_dict().items()}
        if a.mode == 'adaptive':
            p.load_state_dict(best)
        else:
            bestepoch = 5
        p.requires_grad_(False)
        p.eval()
        return dict(before=before, after=diagnose(p, v, A, B, s), selected_epoch=bestepoch, validation_history=scores, loss_history=losses)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        ids = order[(step - 1) * 32:step * 32]
        assert len(ids) == 32
        positions = set(subrng.choice(32, 4, replace=False).tolist())
        cal = [i for j, i in enumerate(ids) if j in positions]
        other = [i for j, i in enumerate(ids) if j not in positions]
        states = {l: [w.detach().clone() for w in weights(mod)[:2]] for l, mod in mods.items()}
        cal_samples, cal_true = collect(m, tok, mods, cap, ds, cal)
        before_params = {l: {k: v.detach().clone() for k, v in p.state_dict().items()} for l, p in pp.items()}
        refresh = {}
        vt = 0.0
        if a.mode in ['adaptive', 'fixed5']:
            tv = time.time()
            validation, _ = collect(m, tok, mods, cap, dev, list(range(16)))
            vt = time.time() - tv
            for l in mods:
                refresh[str(l)] = fit_layer(l, cal_samples[l], validation[l])
        assert all((torch.equal(w, old) for l, mod in mods.items() for w, old in zip(weights(mod)[:2], states[l])))
        other_samples, other_true = collect(m, tok, mods, cap, ds, other)
        diagnostics = {}
        aggregates_pred = []
        aggregates_true = []
        opt.zero_grad(set_to_none=True)
        for l, mod in mods.items():
            A, B, s = weights(mod)
            rows = {r['index']: r for r in cal_samples[l] + other_samples[l]}
            d = pack([rows[i] for i in ids])
            n = int(d['counts'].sum())
            if a.mode == 'oracle':
                A.grad = (cal_true[l][0] + other_true[l][0]) / n
                B.grad = (cal_true[l][1] + other_true[l][1]) / n
                continue
            p = pp[l]
            hd = pack(other_samples[l])
            after = diagnose(p, hd, A, B, s)
            after_state = {k: v.detach().clone() for k, v in p.state_dict().items()}
            p.load_state_dict(before_params[l])
            before = diagnose(p, hd, A, B, s)
            p.load_state_dict(after_state)
            full = diagnose(p, d, A, B, s)
            dg = pred(p, d)
            ga = torch.zeros_like(A)
            gb = torch.zeros_like(B)
            for ii in torch.arange(32, device='cuda').split(8):
                aa, bb = raw_grads(d['x'][ii].flatten(0, 1), dg[ii].flatten(0, 1), A, B, s)
                ga += aa
                gb += bb
            A.grad = ga.detach() / n
            B.grad = gb.detach() / n
            aggregates_pred.extend([A.grad.flatten(), B.grad.flatten()])
            aggregates_true.extend([((cal_true[l][0] + other_true[l][0]) / n).flatten(), ((cal_true[l][1] + other_true[l][1]) / n).flatten()])
            diagnostics[str(l)] = dict(heldout_before=before, heldout_after=after, full_batch=full)
        aggregate_metric = measures(torch.cat(aggregates_pred), torch.cat(aggregates_true)) if aggregates_pred else None
        norm = float(nn.utils.clip_grad_norm_(allparams, 1.0, error_if_nonfinite=True))
        opt.step()
        row = dict(step=step, sample_indices=[int(i) for i in ids], calibration_indices=[int(i) for i in cal], heldout_indices=[int(i) for i in other], supervised_tokens=n, refresh=refresh, diagnostics=diagnostics, all_layer_gradient=aggregate_metric, gradient_norm=norm, seconds=time.time() - t0, validation_collection_seconds=vt)
        hist.append(row)
        write(out / 'history.json', hist)
        print(json.dumps(dict(name=a.name, step=step, seconds=row['seconds'], epochs={l: r['selected_epoch'] for l, r in refresh.items()})), flush=True)
        save = cfg['dev_steps_single']
        if step in save or step == a.steps:
            m.save_pretrained(out / f'step-{step}')
            torch.save({l: dict(config=infos[l]['config'], state={k: v.detach().cpu() for k, v in p.state_dict().items()}) for l, p in pp.items()}, out / f'predictors-step-{step}.pt')
    assert fingerprints(m) == base_before
    if a.mode == 'frozen':
        assert all((torch.equal(v, initial_states[l][k]) for l, p in pp.items() for k, v in p.state_dict().items()))
    torch.save(opt.state_dict(), out / 'lora-optimizer-final.pt')
    cap.close()
    write(out / 'audit.json', dict(complete=True, base_unchanged=True, lora_fixed_during_calibration=True, steps=a.steps, scope=a.scope, architecture=a.architecture, mode=a.mode, elapsed=time.time() - start, validation_excluded_from_training=True))
if __name__ == '__main__':
    main()
