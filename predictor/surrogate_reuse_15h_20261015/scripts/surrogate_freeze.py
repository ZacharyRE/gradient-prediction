"""Explicit recipe/adapter freeze before new candidate-selection readouts."""
from common import *
import argparse

FREEZE=EXP/'configs/surrogate_final_freeze.json'
DECISIONS=EXP/'configs/surrogate_selection_decisions.json'

def entry_key(e):return e['tag']+'__'+e['task']
def entry_signature(e):return hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
def critical_code():return {name:sha(CODE_ROOT/name) for name in ['evaluate.py','common.py','surrogate_freeze.py','freeze_selection_decisions.py','analyze_selection.py']}
def graph_backend_code(root=CODE_ROOT):return {name:sha(root/name) for name in ['graph_teacher.py','teacher_prefix_cache.py','fast_backend.py','research_teacher_cache.py','gpu_features.py','correction4.py','predictor_core.py','reuse_train.py']}

def ensure_surrogate_freeze(args,entries):
    assert entries is not None and FREEZE.exists(),'Freeze all surrogate recipes and candidate adapters before selection readouts'
    frozen=json.loads(FREEZE.read_text());assert frozen['complete']
    assert frozen['code']==critical_code(),'Selection evaluation implementation changed after freeze'
    assert args.max_new_tokens==frozen['max_new_tokens'] and args.model==frozen['model']
    assert args.adapter_group_size==1,'Use the declared serial-adapter protocol'
    for e in entries:
        allowed=frozen['entries'][entry_key(e)]
        assert entry_signature(e)==allowed['entry_sha256'],'Entry changed after freeze'
        assert sha(Path(e['adapter'])/'adapter_model.safetensors')==allowed['adapter_sha256']
        assert sha(data(e['task'],args.split))==frozen['data_sha256'][e['task']][args.split]
    receipt=dict(freeze_sha256=sha(FREEZE),frozen_at=frozen['frozen_at'],purpose=frozen['purpose'])
    if args.split=='selection_holdout':
        assert DECISIONS.exists(),'Freeze all candidate choices before heldout selection outcomes'
        decisions=json.loads(DECISIONS.read_text());assert decisions['freeze_sha256']==sha(FREEZE) and decisions['complete']
        receipt['decisions_sha256']=sha(DECISIONS)
    return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--entries',nargs='+',required=True);ap.add_argument('--recipe-configs',nargs='+',required=True);ap.add_argument('--purpose',required=True);ap.add_argument('--behavior-models',nargs='*',default=[]);a=ap.parse_args()
    assert not FREEZE.exists(),'A final freeze cannot be overwritten'
    for p in (EXP/'results/eval').glob('*/protocol.json'):
        assert json.loads(p.read_text())['args']['split'] not in ['selection_dev','selection_holdout'],'Selection readout exists before freeze'
    entries=[]
    for f in a.entries:entries.extend(json.loads(Path(f).read_text()))
    assert len({entry_key(e) for e in entries})==len(entries)
    graph_used=False;backend_versions={}
    for e in entries:
        config_path=Path(e['adapter']).parent/'config.json'
        if config_path.exists():
            cfg=json.loads(config_path.read_text())
            if cfg.get('teacher_backend')=='graph':
                graph_used=True;root=EXP/'provenance/source_versions'/cfg['source_sha256'];version=graph_backend_code(root)
                assert version==graph_backend_code(),'Selected graph adapter used a different backend implementation'
                backend_versions[cfg['source_sha256']]=version
    backend_audits={}
    if graph_used:
        for name in ['graph_rollout_audit.json','graph_labels_audit.json','graph_flop_accounting_audit.json']:
            path=EXP/'results'/name;assert path.exists() and json.loads(path.read_text())['passed'],f'Require passing graph audit: {name}'
            backend_audits[name]=sha(path)
        long_audits=[]
        for path in (EXP/'results/backend_equivalence').glob('*.json'):
            r=json.loads(path.read_text())
            if r['reference_entry']['steps']>=512 and r.get('reference_teacher_backend')=='graph' and r.get('actual_teacher_backend')=='eager_static' and r['weights_bitwise_equal']:
                root=EXP/'provenance/source_versions'/r['source_sha256']
                if root.exists() and graph_backend_code(root)==graph_backend_code():long_audits.append(dict(path=str(path),sha256=sha(path),backend_code=graph_backend_code(root)))
        assert long_audits,'Require a full512-step graph/eager-static exact-weight audit of the current backend';backend_audits['long_horizon']=long_audits;backend_audits['selected_training_versions']=backend_versions
    config_records={str(Path(p).resolve()):dict(sha256=sha(Path(p)),content=json.loads(Path(p).read_text())) for p in a.recipe_configs}
    records={entry_key(e):dict(entry=e,entry_sha256=entry_signature(e),adapter_sha256=sha(Path(e['adapter'])/'adapter_model.safetensors')) for e in entries}
    tasks=sorted({e['task'] for e in entries})
    record=dict(complete=True,frozen_at=time.time(),purpose=a.purpose,model='1p5b',max_new_tokens=256,full_inputs=True,code=critical_code(),recipes=config_records,entries=records,data_sha256={t:{s:sha(data(t,s)) for s in ['selection_dev','selection_holdout']} for t in tasks},selection='Selection_dev chooses candidate within each predeclared method/start/horizon/family; decisions frozen before selection_holdout; ties lexicographic subset id')
    record['behavior_predictors']={str(Path(p).resolve()):sha(Path(p)) for p in a.behavior_models}
    record['graph_backend_audits']=backend_audits
    record['secondary_analysis_plan_sha256']=sha(EXP/'configs/secondary_analysis_plan.json')
    record['behavior_code']={name:sha(CODE_ROOT/name) for name in ['evaluate_behavior.py','behavior_surrogate.py']}
    with FREEZE.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    log(dict(frozen_entries=len(records),freeze_sha256=sha(FREEZE)))

if __name__=='__main__':main()
