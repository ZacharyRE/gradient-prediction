"""Train one input ablation with the existing bidirectional predictor and loss.

Preload full cached sequences on GPU, matching the original training implementation.
Padding preserves every input token; no input truncation.
"""
from predictor import *
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--source', choices=['x', 'y', 'h', 'xy'], required=True)
    ap.add_argument('--aux', choices=['none', 'mask', 'pos', 'mask_pos'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--preload', action='store_true')
    a = ap.parse_args()
    seed(a.seed)
    out = EXP / 'models' / a.name
    out.mkdir(parents=True, exist_ok=False)
    samples = load_samples(a.cache, 'predictor_train') + load_samples(a.cache, 'predictor_extra')
    dev_samples = load_samples(a.cache, 'predictor_dev')
    if a.source == 'h':
        assert all('h' in s for s in samples + dev_samples), 'H cache must be prepared first'
    # Drop unused H to avoid padding and transferring this large extra tensor.
    if a.source != 'h':
        for s in samples + dev_samples:
            s.pop('h', None)
    meta = torch.load(Path(a.cache) / 'metadata.pt', map_location='cpu', weights_only=False)
    A, B, scale = meta['A'].cuda(), meta['B'].cuda(), meta['scale']
    token_scale = sum(float(s['g'].double().square().sum()) for s in samples)
    token_scale /= sum(s['g'].numel() for s in samples)
    cfg = dict(source=a.source, aux=a.aux, width=512, depth=2, architecture='sequence')
    p = HiddenPredictor(**cfg).cuda()
    p.target_scale.fill_(token_scale ** .5)
    opt = torch.optim.AdamW(p.parameters(), lr=3e-4, weight_decay=.01)
    write(out / 'config.json', dict(args=vars(a), config=cfg, n_train=len(samples),
          n_dev=len(dev_samples), parameters=sum(t.numel() for t in p.parameters()),
          batch_size=32, input_truncation=False, loss='factor-balanced relative A/B MSE + .25 normalized G MSE',
          selection='minimum predictor_dev factor-balanced relative A/B MSE; evaluate every epoch'))
    train_n, dev_n = len(samples), len(dev_samples)
    train_data = pack(samples) if a.preload else None
    dev_data = pack(dev_samples) if a.preload else None
    if a.preload:
        del samples, dev_samples
    # Dedicated order RNG: matching sample order even when XY changes parameter count.
    order_rng = torch.Generator().manual_seed(a.seed)
    best, history, start = float('inf'), [], time.time()
    for epoch in range(1, a.epochs + 1):
        p.train()
        total, n = 0., 0
        for ids in torch.randperm(train_n, generator=order_rng).split(32):
            if a.preload:
                gpu_ids = ids.cuda()
                d = {k:v[gpu_ids] for k,v in train_data.items()}
            else:
                d = pack([samples[i] for i in ids.tolist()])
            ix = torch.arange(len(ids), device='cuda')
            pred = infer(p, d, ix)
            pa, pb = braw(d['x'], pred, A, B, scale)
            with torch.no_grad():
                ta, tb = braw(d['x'], d['g'], A, B, scale)
            la = (pa-ta).square().sum((1,2)) / ta.square().sum((1,2)).clamp_min(1e-12)
            lb = (pb-tb).square().sum((1,2)) / tb.square().sum((1,2)).clamp_min(1e-12)
            lg = (pred-d['g']).square().sum() / d['valid'].sum() / pred.shape[-1] / token_scale
            loss = (la.mean()+lb.mean())/2 + .25*lg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(p.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(ids)
            n += len(ids)
        p.eval()
        with torch.no_grad():
            scores = []
            for st in range(0, dev_n, 16):
                dd = {k:v[st:st+16] for k,v in dev_data.items()} if a.preload else pack(dev_samples[st:st+16])
                result = evaluate(p, dd, meta)
                scores.append((result['selection_score'], len(dd['x'])))
            score = sum(v*k for v,k in scores) / sum(k for _,k in scores)
        row = dict(epoch=epoch, loss=total/n, dev_score=score, seconds=time.time()-start)
        history.append(row)
        write(out/'history.json', history)
        if score < best:
            best = score
            torch.save(dict(config=cfg, state={k:v.detach().cpu() for k,v in p.state_dict().items()},
                            args=vars(a), epoch=epoch, dev_score=score), out/'best.pt')
        print(json.dumps(row), flush=True)
    write(out/'summary.json', dict(best_score=best, elapsed=time.time()-start, complete=True))


if __name__ == '__main__':
    main()
