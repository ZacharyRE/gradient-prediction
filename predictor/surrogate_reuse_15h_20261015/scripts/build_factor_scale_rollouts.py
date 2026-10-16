from common import *
r=json.loads((EXP/'results/factor_scale_fit.json').read_text());assert r['complete'];selected=min(r['results'],key=lambda x:(x['best'],x['q']));path=EXP/'models'/f'mnli_factor_scale_q{selected["q"]}'/'best.pt'
jobs=[dict(name=f'scale_q{selected["q"]}_cv{k}_lr{lr}',kind='cv',k=k,lr=lr,steps=512,save=[32,64,128,256,512],micro=32,diagnostics=False,predictor_path=str(path)) for k in [4,8,16] for lr in [.0003,.001]]
write(EXP/'configs/factor_scale_rollout_jobs.json',jobs);write(EXP/'configs/factor_scale_selection.json',dict(selected=selected,criterion='Smallest predictor_dev centered gradient residual ratio; ties lower PCA dimension',outcome_data_used=False))
