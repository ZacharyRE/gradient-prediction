"""Paired raw/generated targets under native and vLLM inference on the same128."""
import ast
import subprocess
import numpy as np
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
pp=S/'audits/native_target_control_plan.json';plan=json.loads(pp.read_text());ph=sha(pp)
status=json.loads((S/'audits/native_target_control_status.json').read_text())
assert status['all_complete'] and status['plan_sha256']==ph
out=S/'results/native_target_control_analysis.json';assert not out.exists()
names=plan['reused_native_models']+plan['new_models'];assert len(names)==len(set(names))==5
data=read(S/'data/dev.jsonl')[:128];assert len(data)==128 and sha(S/'data/dev.jsonl')==plan['data_sha256']
for name,digest in plan['reused_prediction_sha256'].items():assert sha(S/'results/native/dev'/name/'predictions.jsonl')==digest
replay=json.loads((S/'audits/final_native_early128_replay.json').read_text());assert replay['passed']
assert sha(S/'audits/final_native_early128_replay.json')==plan['early_native_baseline_replay_sha256']
dataset='native_target128';dp=S/f'data/{dataset}.jsonl'
if dp.exists():assert read(dp)==data
else:jsonl(dp,data)
roots={'native':S/'results/native_target_scoring_view/native','vllm':S/'results/native_target_scoring_view/vllm'}
vectors={backend:{'raw':{},'strict':{}} for backend in roots};identities={};manifests={}
for name in names:
    nr=S/'results/native/dev'/name;vr=S/'results/evaluation/dev'/name
    native=read(nr/'predictions.jsonl');full=read(vr/'math_predictions.jsonl');vllm=full[:128]
    assert len(native)==128 and len(full)==500
    assert [r['sample_hash'] for r in native]==[r['sample_hash'] for r in vllm]
    nm=json.loads((nr/'manifest.json').read_text());vm=json.loads((vr/'math_manifest.json').read_text())
    assert nm['script_sha256']==plan['native_script_sha256']
    assert nm['selected_first_n']==128 and nm['batch']==8 and nm['data_sha256']==plan['data_sha256']
    assert nm['model']['files']==vm['model']['files']
    if name!='base':
        assert nm['adapter_loaded_exactly'] and nm['adapter']['files']==vm['adapter']['files']
        if name in plan['artifacts']:
            assert nm['adapter']['files']['adapter_model.safetensors']==plan['artifacts'][name]['weight_sha256']
            assert nm['adapter']['files']['adapter_config.json']==plan['artifacts'][name]['config_sha256']
    manifests[name]={'native':nm,'vllm':vm}
    identities[name]=dict(native_predictions_sha256=sha(nr/'predictions.jsonl'),
        vllm_full_predictions_sha256=sha(vr/'math_predictions.jsonl'),native_manifest_sha256=sha(nr/'manifest.json'),
        vllm_manifest_sha256=sha(vr/'math_manifest.json'))
    for backend,rows in [('native',native),('vllm',vllm)]:
        dest=roots[backend]/name;dest.mkdir(parents=True,exist_ok=True)
        if backend=='native':
            for alias,source in [('math_predictions.jsonl',nr/'predictions.jsonl'),('math_summary.json',nr/'summary.json')]:
                target=dest/alias
                if not target.exists():target.symlink_to(source)
                assert target.resolve()==source.resolve()
        else:
            jsonl(dest/'math_predictions.jsonl',rows)
            write(dest/'math_summary.json',dict(n=128,correct=sum(r['correct'] for r in rows),scope='Exact first128 rows of existing500 outputs; no regeneration/rescoring.'))
        vectors[backend]['raw'][name]=np.array([r['correct'] for r in rows],int)
for name,value in manifests.items():
    for key in ['attention','base_dtype','adapter_dtype','compute_autocast','use_cache','temperature','max_new_tokens']:
        assert value['native'][key]==manifests['base']['native'][key]
    for key in ['schema','engine','batch_invariant','max_tokens','temperature','chunk_size','data_sha256']:
        assert value['vllm'][key]==manifests['base']['vllm'][key]
warnings=[];audit_records={}
for backend,root in roots.items():
    tag='native_target128_'+backend;ap=S/f'audits/scoring_{tag}.json'
    if not ap.exists():
        with (S/f'logs/strict_{tag}.log').open('w') as log:
            subprocess.run([sys.executable,str(S/'scripts/score_sensitivity.py'),'--dataset',dataset,
                '--prediction-root',str(root),'--tag',tag,'--models',*names],stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=max(1,DEADLINE-time.time()))
    a=json.loads(ap.read_text());assert a['script_sha256']==plan['strict_script_sha256']
    assert a['dataset_sha256']==sha(dp) and set(a['models'])==set(names)
    for name in names:
        assert a['models'][name]['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
        vectors[backend]['strict'][name]=np.array(a['models'][name]['strict_vector'],int)
    for j,w in enumerate(a['parser_comparison_warnings']):
        i=w['index'];item=dict(case_id=f'{tag}:{j}',backend=backend,warning=w,
            problem=data[i]['problem'],solution=data[i]['solution'])
        if 'model' in w:
            name=w['model'];r=read(root/name/'math_predictions.jsonl')[i]
            item.update(model=name,prediction=r['prediction'],original_correct=r['correct'],
                        strict_correct=bool(a['models'][name]['strict_vector'][i]),
                        predictions_sha256=sha(root/name/'math_predictions.jsonl'))
        warnings.append(item)
    audit_records[backend]=dict(audit_sha256=sha(ap),warning_count=len(a['parser_comparison_warnings']),judgments=640)
jsonl(S/'audits/native_target_control_warning_cases.jsonl',warnings)
source=S/'scripts/analyze_confirmation.py'
primary=json.loads((S/'results/confirmation_plan.json').read_text());assert sha(source)==primary['analysis_script_sha256']
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
compare=namespace['compare'];results={}
for mode in ['raw','strict']:
    backend_results={}
    for backend,values in vectors.items():
        vv=values[mode];b=vv['base']
        backend_results[backend]=dict(base_correct=int(b.sum()),
            models={name:dict(correct=int(v.sum()),vs_own_base=compare((v-b)[None,:])) for name,v in vv.items()},
            pairs=[dict(generated=g,raw=r,generated_correct=int(vv[g].sum()),raw_correct=int(vv[r].sum()),
                        generated_minus_raw=compare((vv[g]-vv[r])[None,:])) for g,r in plan['pairs']])
    interaction=[]
    for g,r in plan['pairs']:
        dn=vectors['native'][mode][g]-vectors['native'][mode][r]
        dv=vectors['vllm'][mode][g]-vectors['vllm'][mode][r]
        interaction.append(dict(generated=g,raw=r,native_minus_vllm_target_effect=compare((dn-dv)[None,:])))
    results[mode]=dict(backends=backend_results,backend_target_interactions=interaction)
alignment_path=S/'audits/loss_accuracy_alignment.json';alignment=json.loads(alignment_path.read_text())
assert alignment['ce_indices_in_dev']==list(range(64)) and alignment['dev_sha256']==plan['data_sha256']
ce_rows=[];initial_losses=[]
for pair in alignment['pairs']:
    for kind in ['raw','generated']:
        record=pair[kind];name=record['name']+f"_epoch{record['epoch']}";assert name in names
        assert identities[name]['vllm_full_predictions_sha256']==record['prediction_sha256']
        assert sha(S/'results/training'/record['name']/'history.json')==record['history_sha256']
        initial_losses.append(record['initial_raw_reference_ce'])
        ce_rows.append(dict(model=name,kind=kind,reference_ce=record['endpoint_raw_reference_ce'],
            reference_ce_change=record['raw_reference_ce_change'],
            native_correct_same64=int(vectors['native']['raw'][name][:64].sum()),
            native_strict_correct_same64=int(vectors['native']['strict'][name][:64].sum()),
            vllm_correct_same64=int(vectors['vllm']['raw'][name][:64].sum()),
            vllm_strict_correct_same64=int(vectors['vllm']['strict'][name][:64].sum())))
assert len(set(initial_losses))==1
ce_alignment=dict(n=64,source_audit_sha256=sha(alignment_path),base_reference_ce=initial_losses[0],
    base_native_correct=int(vectors['native']['raw']['base'][:64].sum()),
    base_vllm_correct=int(vectors['vllm']['raw']['base'][:64].sum()),rows=ce_rows,
    scope='Both referenceCE and nativefreegeneration useHF/PEFT BF16/SDPAmath withFP32adapters onexactlythe same64questions/checkpoints. Teacherforcing versusautoregression/batch/cache still differ; no purekernel equivalenceclaim.')
write(out,dict(time=time.time(),script_sha256=sha(__file__),plan_sha256=ph,n=128,seed=43,
    identities=identities,scoring_audits=audit_records,warning_count=len(warnings),
    warning_cases_sha256=sha(S/'audits/native_target_control_warning_cases.jsonl'),results=results,
    reference_ce_alignment_same_native64=ce_alignment,
    scope=plan['scope']+' NativeversusvLLM jointly changesbackend,adapterstorage/arithmeticdetails; '
          'an interaction cannot isolateonekernelordtype. Only questionpairedCIis meaningfulforonesingletrainingseed.'))
print(json.dumps(results,indent=2))
