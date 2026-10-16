"""Training-only two-sided gradient-matrix basis and compact fitting targets."""
from predictor_core import *
from factor_surrogate import FactorSurrogate
from updates_legacy import clock
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='critic_v1');ap.add_argument('--name',default='factor_v1');a=ap.parse_args();gpu_guard();guard_space();seed(919)
    source=EXP/'cache'/a.source;out=EXP/'cache'/a.name;out.mkdir(exist_ok=True)
    manifest=json.loads((source/'complete.json').read_text());states=[];tick=clock()
    for item in manifest['states']:
        name=item['state']['name'];path=source/name;meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False)
        states.append(dict(name=name,train=load_samples(path/'predictor_train'),dev=load_samples(path/'predictor_dev'),meta=meta))
    dim=states[0]['meta']['A'].shape[1];cov_u=torch.zeros(dim,dim,device='cuda',dtype=torch.float64);cov_v=torch.zeros_like(cov_u);basis_n=0
    for state in states:
        rows=state['train'];ids=list(map(int,np.random.default_rng(919).choice(len(rows),min(64,len(rows)),replace=False)))
        state_u=torch.zeros_like(cov_u);state_v=torch.zeros_like(cov_v)
        with torch.no_grad():
            for ix in sample_chunks(rows,ids,max_n=4,budget=4096):
                d=pack([rows[i] for i in ix]);C=d['g'].transpose(-1,-2)@d['x'].float()
                state_u+=(C@C.transpose(-1,-2)).sum(0).double();state_v+=(C.transpose(-1,-2)@C).sum(0).double();basis_n+=len(ix)
        energy=state_u.trace().clamp_min(1e-20);cov_u+=state_u/energy;cov_v+=state_v/energy
    eu,U=torch.linalg.eigh(cov_u);ev,V=torch.linalg.eigh(cov_v);U=U.flip(1)[:,:128].float();V=V.flip(1)[:,:256].float()
    assert torch.allclose(U.T@U,torch.eye(128,device='cuda'),atol=1e-4,rtol=1e-4)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True);tokens=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABC'];assert all(len(tok.encode(c,add_special_tokens=False))==1 for c in 'ABC')
    prototype=FactorSurrogate(dim,128,256,class_tokens=tokens).cuda();prototype.U.copy_(U);prototype.V.copy_(V)
    basis_seconds=clock()-tick;torch.save(dict(U=U.cpu(),V=V.cpu(),class_tokens=tokens),out/'basis.pt');records=[]
    reserve_worker_cache()
    for state in states:
        meta=state['meta'];A=meta['A'].cuda();B=meta['B'].cuda();s=meta['scale'];saved={};start=clock()
        for split in ['train','dev']:
            rows=state[split];parts={k:[] for k in ['features','classes','coefficients','ga','gb']}
            with torch.no_grad():
                for ix in sample_chunks(rows,max_n=16,budget=4096):
                    d=pack([rows[i] for i in ix]);features,cls=prototype.features(d);coeff=(d['g']@U).transpose(-1,-2)@(d['x'].float()@V);ga,gb=raw(d['x'],d['g'],A,B,s)
                    for k,v in [('features',features),('classes',cls),('coefficients',coeff),('ga',ga),('gb',gb)]:parts[k].append(v.cpu())
            saved[split]={k:torch.cat(v) for k,v in parts.items()}
        path=out/f'{state["name"]}.pt';torch.save(dict(meta=meta,**saved),path)
        records.append(dict(name=state['name'],file=path.name,train_n=len(state['train']),dev_n=len(state['dev']),seconds=clock()-start,sha256=sha(path)));log(records[-1])
    write(out/'complete.json',dict(complete=True,source_cache=str(source),source_sha256=sha(source/'complete.json'),states=records,basis_training_examples=basis_n,basis_seconds=basis_seconds,total_seconds=clock()-tick,basis_energy_u128=float(eu[-128:].sum()/eu.sum()),basis_energy_v256=float(ev[-256:].sum()/ev.sum()),class_tokens=tokens,source_code_sha256=source_signature(__file__),basis_uses_predictor_train_only=True,targets='predictor_train for fitting and predictor_dev for fit selection; no downstream evaluation data',basis_state_weighting='Each teacher state covariance normalized by its own trace before aggregation'))

if __name__=='__main__':main()
