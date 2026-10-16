"""Compare frozen surrogates on identical held-out gradients at fixed teacher states.

Diagnostic only: no fitting, checkpoint selection, or downstream-test readout.
Post-hoc coefficients are descriptive and never applied to a training run.
"""
from predictor_core import *
import predictor_core as pc
import state_features as sf
from fast_backend import configure
from safetensors.torch import load_file
import argparse,gc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['paws','mnli'],required=True);a=ap.parse_args();task=a.task
    source=Path(json.loads((EXP/'configs/protocol.json').read_text())['source_study'])
    warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=101);A,B,s=weights(mod);cap=Capture(mod,m);configure(True,8192);sf.install()
    ds=Dataset(task,'gradient_test',tok)
    suffix='_globalfit' if task=='mnli' else ''
    candidates=[source/'models'/f'{task}_predictor{suffix}_p123'/'best.pt']
    candidates+=sorted((EXP/'models').glob(f'{task}__refit*_p123/best.pt'))
    candidates+=sorted((EXP/'models').glob(f'{task}__multistate*_p123/best.pt'))
    rows=[];receipts=[]
    for step in [0,32,128,256]:
        adapter=Path(warm['adapter']) if step==0 else EXP/'models'/f'{task}__fast_oracle32_u101_p123'/f'step-{step}'
        values=load_file(str(adapter/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in values.items() if k.endswith('lora_A.weight')))
            B.copy_(next(v for k,v in values.items() if k.endswith('lora_B.weight')))
        samples,_,audit=collect(m,tok,mod,cap,ds,list(range(len(ds))),micro=32,keep=True,verify_reconstruction=True)
        assert audit['min_reconstruction_cosine']>.999 and audit['max_reconstruction_relative_l2']<.02
        receipts.append(dict(teacher_step=step,adapter_sha256=sha(adapter/'adapter_model.safetensors'),collection_audit=audit))
        for path in candidates:
            cfg=json.loads((path.parent/'config.json').read_text())
            explicit=cfg.get('job',{}).get('explicit_state',False)
            p=sf.load_state_checkpoint(path,A,B,s)[0] if explicit else load_predictor(path)[0]
            report,pred,truth,_=diagnose(p,samples,A,B,s,return_grads=True,factor_detail=True)
            metrics={};cut=A.numel()
            for name,pp,tt in [('all',pred,truth),('A',pred[:,:cut],truth[:,:cut]),('B',pred[:,cut:],truth[:,cut:])]:
                pp=pp-pp.mean(0);tt=tt-tt.mean(0);cov=float((pp*tt).sum());pv=float(pp.square().sum());tv=float(tt.square().sum())
                coefficient=cov/max(pv,1e-30)
                metrics[name]=dict(centered_correlation=cov/max((pv*tv)**.5,1e-30),unit_coefficient_residual_ratio=float((tt-pp).square().sum())/max(tv,1e-30),posthoc_optimal_coefficient=coefficient,posthoc_min_residual_ratio=float((tt-coefficient*pp).square().sum())/max(tv,1e-30))
            row=dict(task=task,teacher_step=step,predictor=path.parent.name,predictor_sha256=sha(path),explicit_state=explicit,n=len(samples),metrics=metrics,batch_lora=report['batch_lora'])
            rows.append(row);log(dict(task=task,step=step,predictor=path.parent.name,residual_ratio=metrics['all']['unit_coefficient_residual_ratio']))
            del p,pred,truth;gc.collect()
        write(EXP/'results'/f'surrogate_transfer_{task}.json',dict(rows=rows,teacher_receipts=receipts,gradient_data_sha256=sha(data(task,'gradient_test')),full_inputs=True,scope='All frozen predictors evaluated on the same held-out gradient examples and same Oracle32 LR3e-4 seed101 states. No fitting or checkpoint selection uses these diagnostics. Post-hoc optimal coefficients use diagnostic labels and are never deployed; they quantify possible rescaling only. These audits are research overhead, excluded from adaptation timing.'))
        del samples;gc.collect()
    cap.close()


if __name__=='__main__':main()
