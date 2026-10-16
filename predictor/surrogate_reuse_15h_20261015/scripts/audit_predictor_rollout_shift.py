"""Measure gradient fidelity along actual development rollouts, without fitting."""
from predictor_core import *
from fit_critic import evaluate_state
from gpu_features import collect_gpu
from fast_backend import configure
from updates_legacy import clock
from safetensors.torch import load_file
import argparse,gc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--states',required=True);ap.add_argument('--predictors',nargs='+',required=True);ap.add_argument('--name',default='predictor_rollout_shift');a=ap.parse_args()
    states=json.loads(Path(a.states).read_text());assert all(st['warmup_seed'] in [201,301,311,321] for st in states),'Development states only'
    start=time.time();m,tok,mod=setup('1p5b',states[0]['adapter'],rng=913);cap=Capture(mod,m);A,B,s=weights(mod);configure(True,8192)
    ds=Dataset('mnli','predictor_dev',tok);models=[(path,load_predictor(path)[0]) for path in a.predictors];result=[]
    for state in states:
        if deadline(5400):raise RuntimeError('Reserve final validation time')
        vals=load_file(str(Path(state['adapter'])/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in vals.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in vals.items() if k.endswith('lora_B.weight')))
        tick=clock();rows,truth,audit=collect_gpu(m,tok,mod,cap,ds,list(range(len(ds))),micro=16,keep=True,verify_reconstruction=True);label_seconds=clock()-tick
        metrics={}
        for path,p in models:
            tick=clock();metrics[path]=dict(**evaluate_state(p,rows,A,B,s),evaluation_seconds=clock()-tick)
        result.append(dict(state=state,adapter_sha256=sha(Path(state['adapter'])/'adapter_model.safetensors'),label_seconds=label_seconds,gradient_reconstruction=audit,predictors=metrics))
        write(EXP/'results'/f'{a.name}.json',dict(complete=False,rows=result));log(dict(state=state['name'],metrics=metrics));del rows,truth;gc.collect()
    cap.close();write(EXP/'results'/f'{a.name}.json',dict(complete=True,rows=result,wall_seconds=time.time()-start,n=len(ds),data_sha256=sha(data('mnli','predictor_dev')),predictor_sha256={path:sha(path) for path,_ in models},states_sha256=sha(a.states),full_inputs=True,source_sha256=source_signature(__file__),scope='Development-only gradient diagnostic on predictor_dev. No model fitting, no selection/final readouts. Compares actual Oracle/CV trajectories at matched update counts; gradient ratios are not downstream performance or timing claims.'))

if __name__=='__main__':main()
