from predictor_core import *
from legacy_cpu_diagnostic import pack_cpu,covariance_summary
from scipy.stats import spearmanr

@torch.no_grad()
def main():
    torch.set_num_threads(4);source=Path(json.loads((EXP/'configs/protocol.json').read_text())['source_study']);results={}
    for task in ['paws','mnli']:
        suffix='_globalfit' if task=='mnli' else ''
        obj=torch.load(source/'models'/f'{task}_predictor{suffix}_p123/best.pt',map_location='cpu',weights_only=False)
        p=Predictor(**obj['config']);p.load_state_dict(obj['state']);p.eval()
        cache=source/'cache'/task;rows=load_samples(cache/'predictor_dev');meta=torch.load(cache/'metadata.pt',map_location='cpu',weights_only=False)
        A,B,s=meta['A'],meta['B'],meta['scale'];pp=[];tt=[]
        for ix in sample_chunks(rows,max_n=4,budget=2048):
            d=pack_cpu([rows[i] for i in ix]);pg=raw(d['x'],p(d),A,B,s);tg=raw(d['x'],d['g'],A,B,s)
            pp.append(torch.cat([v.flatten(1) for v in pg],1));tt.append(torch.cat([v.flatten(1) for v in tg],1))
        pp=torch.cat(pp).double();tt=torch.cat(tt).double();N=len(pp)
        pn=pp.norm(dim=1);tn=tt.norm(dim=1)
        out=dict(n=N,norm_spearman=float(spearmanr(pn,tn).statistic),norm_log_pearson=float(torch.corrcoef(torch.stack([pn.clamp_min(1e-20).log(),tn.clamp_min(1e-20).log()]))[0,1]),covariance=covariance_summary(pp,tt))
        for lam in [0.,1.,3.]:
            residual=tt-lam*pp;energy=residual.square().sum(1);sum_energy=float(residual.sum(0).square().sum());uniform=float(N*energy.sum())-sum_energy
            for label,weight in [('pred_norm',pn),('true_norm_unavailable_bound',energy.sqrt())]:
                prob=.5/N+.5*weight/weight.sum()
                variance=float((energy/prob).sum())-sum_energy
                out[f'lambda{lam:g}_{label}_variance_vs_uniform']=variance/uniform
        sources=read(source/'data'/task/'predictor_dev.jsonl');gold=[sources[r['index']]['gold'] for r in rows]
        within=0.;centered=(tt-tt.mean(0)).square().sum()
        for label in set(gold):
            block=tt[torch.tensor([g==label for g in gold])];within+=float((block-block.mean(0)).square().sum())
        out['within_label_variance_fraction']=within/float(centered)
        results[task]=out;log(dict(task=task,**out))
    assert not torch.cuda.is_initialized()
    write(EXP/'results/cpu_sampling_diagnostic.json',dict(results=results,scope='Existing predictor_dev at warmup only; exploratory mechanism diagnostics, no final data',cuda_initialized=False))

if __name__=='__main__':main()
