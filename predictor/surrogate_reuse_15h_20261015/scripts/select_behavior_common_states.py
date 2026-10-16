"""Score every behavior candidate on the same development states and normalization."""
from fit_anchored_behavior import training_features
from predictor_core import *
from behavior_surrogate import BehaviorSurrogate,AnchoredBehaviorSurrogate
from updates_legacy import clock
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());gpu_guard();guard_space()
    started=clock();cache,anchor_indices,tokens,prep,source,states=training_features(plan['cache'])
    train=cache['predictor_train'];dev=cache['predictor_dev'];y=train['logits'];dy=dev['logits'];x=dev['features'];initial=x[anchor_indices];iy=dy[anchor_indices]
    norm=y.square().mean().clamp_min(1e-8);change_scale=(y-y.mean(0,keepdim=True)).square().mean().clamp_min(1e-8)
    def metrics(pred):
        mse=float((pred-dy).square().mean()/norm);change=float(((pred-pred.mean(0,keepdim=True))-(dy-dy.mean(0,keepdim=True))).square().mean()/change_scale)
        return dict(normalized_logit_mse=mse,normalized_state_change_mse=change,joint_score=mse+change,teacher_class_agreement=float((pred.argmax(-1)==dy.argmax(-1)).float().mean()),student_accuracy=float((pred.argmax(-1)==dev['classes']).float().mean()))
    result=[]
    for spec in plan['models']:
        path=Path(spec['path']);obj=torch.load(path,map_location='cuda',weights_only=False);ctor=AnchoredBehaviorSurrogate if obj['config'].get('kind')=='anchored_behavior' else BehaviorSurrogate;p=ctor(**obj['config']).cuda();p.load_state_dict(obj['state_dict']);p.eval()
        with torch.no_grad():
            pred=p(x,initial,iy) if getattr(p,'requires_initial_teacher',False) else p(x if p.delta_features else x[...,:3*p.dim])
        result.append(dict(**spec,model=path.parent.name,predictor_sha256=sha(path),metrics=metrics(pred)));del p
    eligible=[r for r in result if r['eligible']];assert eligible
    selected=min(eligible,key=lambda r:(r['metrics']['joint_score'],r['model']))
    write(EXP/'results/behavior_common_selection.json',dict(complete=True,selected=selected,rows=result,initial_teacher_baseline=metrics(iy),feature_preparation_seconds=prep,wall_seconds=clock()-started,cache_sha256=sha(source/'complete.json'),plan_sha256=sha(a.plan),states=[st['name'] for st in states],source_sha256=source_signature(__file__),scope='All models use identical12 development states,129 predictor_dev examples and common training-derived normalizers. Select at most one eligible behavior model; no selection_dev/holdout outcome used. Coverage-specific fit scores are not directly compared.'))
    log(dict(selected=selected['model'],joint_score=selected['metrics']['joint_score'],initial_baseline=metrics(iy)))

if __name__=='__main__':main()
