"""Fresh five-candidate selection workflow, verified against already frozen choices."""
import time as _time
SCRIPT_START=_time.perf_counter()
from common import *
from surrogate_freeze import FREEZE,DECISIONS,entry_key
import argparse,sys

def wait_for_context_exit():
    holder=reservation_state();assert holder is not None
    deadline_at=time.monotonic()+30
    while True:
        text=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        pids=[int(x.strip()) for x in text.splitlines() if x.strip()]
        if all(pid==holder['pid'] for pid in pids):return
        if time.monotonic()>=deadline_at:raise RuntimeError('GPU context did not exit; refuse the next pipeline process')
        time.sleep(.25)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True);ap.add_argument('--family',choices=['random','genre'],required=True);ap.add_argument('--repeat',type=int,required=True);a=ap.parse_args()
    frozen=json.loads(FREEZE.read_text());decisions=json.loads(DECISIONS.read_text());assert decisions['complete'] and (EXP/'results/final_selection_analysis.json').exists(),'Run benchmark only after scientific readouts/choices are complete'
    policy_path=EXP/'configs/surrogate_cost_policy.json';assert frozen['recipes'][str(policy_path)]['sha256']==sha(policy_path);plan=json.loads(policy_path.read_text())['measured_selection_pipeline_plan'];condition=next(c for c in plan['conditions'] if c['name']==a.condition);assert a.family in plan['families'] and 1<=a.repeat<=plan['repetitions']
    warm=plan['warmup_seed'];d=[d for d in decisions['decisions'] if d['warmup_seed']==warm and d['family']==a.family and d['target']==plan['target'] and all(d[k]==condition[k] for k in ['method','steps','selector'])];assert len(d)==1;decision=d[0]
    bytag={r['entry']['tag']:r['entry'] for r in frozen['entries'].values()};entries=sorted([bytag[x['tag']] for x in decision['candidate_scores']],key=lambda e:e['subset']);assert len(entries)==5
    run_id=f'selection_pipeline_{a.family}_{a.condition}_r{a.repeat}_{time.time_ns()}';out=EXP/'results/selection_pipeline_work'/run_id;out.mkdir(parents=True);entries_path=out/'entries.json';write(entries_path,entries);stages=[];training_receipts=[]
    def child(label,script,args):
        wait_for_context_exit();tick=time.perf_counter();log_path=out/f'{label}.log'
        with log_path.open('w') as log_file:
            subprocess.run([sys.executable,str(CODE_ROOT/script),*map(str,args)],env=dict(os.environ,BENCH_FINAL_VALIDATION='1'),stdout=log_file,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        wait_for_context_exit();seconds=time.perf_counter()-tick;stages.append(dict(name=label,seconds=seconds,log=str(log_path)));return seconds
    def train(e,label):
        before=set((EXP/'results/pipeline_training').glob('*.json'));child(label,'replay_reuse.py',['--reference-adapter',e['adapter'],'--repeat',a.repeat,'--measurement-scope','selection_pipeline'])
        added=set((EXP/'results/pipeline_training').glob('*.json'))-before;assert len(added)==1;path=added.pop();receipt=json.loads(path.read_text());assert receipt['weights_bitwise_equal'] and receipt['reference_entry']==e and receipt['measurement_scope']=='selection_pipeline';training_receipts.append(dict(path=str(path),sha256=sha(path),reference_tag=e['tag'],actual_adapter=receipt['actual_adapter'],weights_bitwise_equal=True))
    for i,e in enumerate(entries):train(e,f'candidate_{i}_training')
    selector=condition['selector'];name=run_id+'_readout'
    if selector=='generated_accuracy':
        child('selection_readout','evaluate.py',['--model','1p5b','--entries',entries_path,'--name',name,'--split','selection_dev','--max-new-tokens',256,'--fresh-generation']);readout_root=EXP/'results/eval'/name;reference_root=EXP/'results/eval/surrogate_final_selection_dev';model=None
    elif selector.startswith('exact_teacher_'):
        child('selection_readout','evaluate_forced_teacher.py',['--entries',entries_path,'--name',name,'--split','selection_dev']);readout_root=EXP/'results/behavior'/name;reference_root=EXP/'results/behavior/surrogate_final_behavior_selection_dev';model='teacher'
    else:
        model=selector.split(':')[0];predictor=[p for p in frozen['behavior_predictors'] if Path(p).parent.name==model];assert len(predictor)==1
        child('selection_readout','evaluate_behavior.py',['--entries',entries_path,'--predictors',predictor[0],'--name',name,'--split','selection_dev']);readout_root=EXP/'results/behavior'/name;reference_root=EXP/'results/behavior/surrogate_final_behavior_selection_dev'
    scores=[];readout_checks=[]
    for e in entries:
        filename=entry_key(e)+'.jsonl' if model is None else e['tag']+'__'+model+'.jsonl';actual=read(readout_root/filename);reference=read(reference_root/filename)
        assert len(actual)==len(reference) and all(x['sample_id']==y['sample_id'] and x['gold']==y['gold'] for x,y in zip(actual,reference))
        if model is None:
            assert actual==reference,'Fresh generated readout differs from frozen records';score=float(np.mean([r['correct'] for r in actual]))
        else:
            logits=np.array([r['logits'] for r in actual]);expected=np.array([r['logits'] for r in reference]);assert np.array_equal(logits,expected),'Fresh forced logits differ from frozen readout'
            prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);score=float(prob[np.arange(len(actual)),['ABC'.index(r['gold']) for r in actual]].mean())
        scores.append(dict(subset=e['subset'],tag=e['tag'],score=score));readout_checks.append(dict(tag=e['tag'],fresh_sha256=sha(readout_root/filename),reference_sha256=sha(reference_root/filename),predictions_equal=True))
    scores.sort(key=lambda r:(-r['score'],r['subset']));assert scores==decision['candidate_scores'] and scores[0]['subset']==decision['selected_subset'],'Fresh selection differs from frozen choice'
    spec=json.loads((EXP/'configs/surrogate_analysis_spec.json').read_text());reference_selector=spec.get('oracle_reference_by_population',{}).get('candidate',spec['oracle_reference']);ref=[e for e in bytag.values() if e.get('warmup_seed')==warm and e.get('subset')==scores[0]['subset'] and all(e.get(k)==v for k,v in reference_selector.items())];assert len(ref)==1;ref=ref[0]
    already=any(e['adapter_sha256']==ref['adapter_sha256'] and e['method']==ref['method'] and e['steps']==ref['steps'] for e in entries)
    if not already:train(ref,'chosen_reference_oracle_training')
    elapsed=time.perf_counter()-SCRIPT_START
    record=dict(complete=True,condition=a.condition,family=a.family,repeat=a.repeat,warmup_seed=warm,method=condition['method'],steps=condition['steps'],selector=selector,selected_subset=decision['selected_subset'],pipeline_script_seconds=elapsed,sum_child_process_seconds=sum(s['seconds'] for s in stages),stages=stages,training_receipts=training_receipts,readout_checks=readout_checks,fresh_predictions_match=True,selected_candidate_matches=True,final_reference_already_trained=already,final_reference_tag=ref['tag'],freeze_sha256=sha(FREEZE),decisions_sha256=sha(DECISIONS),cost_policy_sha256=sha(policy_path),readout_protocol_sha256=sha(readout_root/'protocol.json'),scope=plan['scope'],qualification=plan['qualification'],artifact_aliasing='Readout opens frozen reference adapter paths after every newly simulated adapter has passed bitwise tensor equality. Predictions are freshly computed; no saved outcomes are reused.',source_sha256=source_signature(__file__))
    write(EXP/'results/selection_pipeline'/f'{run_id}.json',record);log({k:record[k] for k in ['complete','condition','family','repeat','pipeline_script_seconds','selected_subset','final_reference_already_trained']})

if __name__=='__main__':main()
