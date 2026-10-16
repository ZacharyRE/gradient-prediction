from research import *
from state_features import augment,install,state_collect,state_pack,delta_output

def main():
    warm=json.loads((EXP/'configs/paws_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);A,B,s=weights(mod);cap=Capture(mod);ds=Dataset('paws','update',tok)
    rows=features(m,tok,mod,cap,ds,[0,1,2,3]);base,_=load_predictor(predictor_path('paws',123));p,_=load_predictor(predictor_path('paws',123))
    p=augment(p,A,B,s,A.detach().clone(),B.detach().clone(),dict(delta_gain=100,cal_scope='delta_head'));install()
    with torch.no_grad():
        d=state_pack(rows);initial=float((base(d)-p(d)).abs().max())
        assert initial==0,initial
        B.add_(.001*torch.randn_like(B))
    labels,_,_=state_collect(m,tok,mod,cap,ds,[0,1,2,3],keep=True,verify_reconstruction=False)
    stored=labels[0]['delta_y'].clone()
    with torch.no_grad():B.add_(.001*torch.randn_like(B))
    assert torch.equal(state_pack(labels)['delta_y'][0,:len(stored)].cpu(),stored)
    current=delta_output(labels[0]['x'].cuda(),p).cpu()
    assert float((current-stored).norm())>0
    result=dict(passed=True,initial_prediction_max_absolute_difference=initial,replay_uses_captured_state=True,state_change_norm=float((current-stored).norm()))
    write(EXP/'results/state_feature_audit.json',result);log(result);cap.close()

if __name__=='__main__':main()
