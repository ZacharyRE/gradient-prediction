"""Optional fixed training-energy normalization; primary fitting is unchanged."""
from predictor_core import *

@torch.no_grad()
def estimate_factor_normalizers(rows,A,B,s):
    energy=torch.zeros(2,dtype=torch.float64,device=A.device);seen=0
    for ix in sample_chunks(rows):
        d=pack([rows[i] for i in ix]);ta,tb=raw(d['x'],d['g'],A,B,s)
        energy+=torch.stack([t.square().sum((1,2)).double().sum() for t in [ta,tb]])
        seen+=len(ix)
    assert seen==len(rows) and seen>0
    values=(energy/seen).clamp_min(1e-12).tolist()
    assert all(np.isfinite(v) and v>0 for v in values)
    return dict(factor_A=values[0],factor_B=values[1],n=seen,floor=1e-12,
                meaning='Training mean squared Frobenius norm of each raw LoRA factor gradient; fixed across examples and epochs')

def objective_global(p,d,A,B,s,*,normalizers,aggregate_weight=0):
    assert aggregate_weight==0,'This isolated objective has no extra batch-loss term'
    g=p(d);pa,pb=raw(d['x'],g,A,B,s)
    with torch.no_grad():ta,tb=raw(d['x'],d['g'],A,B,s)
    la=(pa-ta).square().sum((1,2)).mean()/normalizers['factor_A']
    lb=(pb-tb).square().sum((1,2)).mean()/normalizers['factor_B']
    return (la+lb)/2+.25*(g-d['g']).square().sum()/d['valid'].sum()/g.shape[-1]/p.target_scale.square()

@torch.no_grad()
def diagnose_global(p,rows,A,B,s,*,normalizers):
    result,pp,tt,_=diagnose(p,rows,A,B,s,return_grads=True)
    cut=A.numel();error=(pp.double()-tt.double()).square()
    result['global_individual_score']=float(.5*(error[:,:cut].sum(1).mean()/normalizers['factor_A']+
                                               error[:,cut:].sum(1).mean()/normalizers['factor_B']))
    return result
