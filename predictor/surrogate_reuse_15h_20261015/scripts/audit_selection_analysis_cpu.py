"""Synthetic integration audit: dev chooses a bad candidate; holdout must not reselect."""
from common import *
import tempfile,sys

def main():
    with tempfile.TemporaryDirectory(prefix='surrogate-analysis-audit-') as temp:
        root=Path(temp);(root/'configs').mkdir();(root/'results').mkdir();(root/'data/mnli').mkdir(parents=True);starts=[302,303,304];n=12
        data_rows=[dict(sample_id=f'example{i}',gold='ABC'[i%3],group_id=f'group{i//3}',genre='fiction') for i in range(n)]
        for split in ['selection_dev','selection_holdout']:jsonl(root/'data/mnli'/f'{split}.jsonl',data_rows)
        spec=dict(heldout_warmups=starts,oracle_reference=dict(method='oracle',kind='oracle',steps=128,lr=.0003),oracle_reference_by_population=dict(candidate=dict(method='oracle',kind='oracle',steps=128,lr=.0003),fullpool=dict(method='full_oracle',kind='oracle',steps=256,lr=.001)),random_family_train_all_control=dict(method='full_oracle',kind='oracle',steps=256,lr=.001,subset='update'),direct_paired_controls=[dict(family='random',actual=dict(method='cv',steps=128),reference=dict(method='oracle',steps=128),purpose='Synthetic paired audit')]);specpath=root/'configs/surrogate_analysis_spec.json';write(specpath,spec)
        entries={};dev={};hold={};decisions=[]
        def add(e,devcount,holdcount):
            key=e['tag']+'__mnli';entries[key]=dict(entry=e);dev[key]=np.arange(n)<devcount;hold[key]=np.arange(n)<holdcount
        for w in starts:
            add(dict(tag=f'w{w}',task='mnli',warmup_seed=w,steps=0),6,6)
            add(dict(tag=f'full{w}',task='mnli',warmup_seed=w,subset='update',method='full_oracle',kind='oracle',steps=256,lr=.001),11,11)
            for method,kind in [('oracle','oracle'),('cv','cv')]:
                scores=[]
                for i in range(5):
                    tag=f'{method}_{w}_{i}';dc=10-i if method=='oracle' else 6+i;add(dict(tag=tag,task='mnli',warmup_seed=w,subset=f'candidate_{i}',method=method,kind=kind,steps=128,lr=.0003),dc,10-i if method=='oracle' else 6+i);scores.append(dict(tag=tag,subset=f'candidate_{i}',score=dc/n))
                scores.sort(key=lambda x:(-x['score'],x['subset']));decisions.append(dict(family='random',warmup_seed=w,update_seed=w+99,method=method,kind=kind,steps=128,lr=.0003,loss_objective='full_answer',true_every=None,target='all',selector='generated_accuracy',selected_subset=scores[0]['subset'],selected_tag=scores[0]['tag'],candidate_scores=scores))
        frozen=dict(complete=True,entries=entries,recipes={str(specpath):dict(sha256=sha(specpath))},code={name:sha(CODE_ROOT/name) for name in ['evaluate.py','common.py','surrogate_freeze.py','freeze_selection_decisions.py','analyze_selection.py']},data_sha256={'mnli':{split:sha(root/'data/mnli'/f'{split}.jsonl') for split in ['selection_dev','selection_holdout']}});freeze=root/'configs/surrogate_final_freeze.json';write(freeze,frozen);decisionpath=root/'configs/surrogate_selection_decisions.json';write(decisionpath,dict(complete=True,freeze_sha256=sha(freeze),decisions=decisions))
        tags=sorted(entries);metadata=dict(freeze_sha256=sha(freeze),decision_sha256=sha(decisionpath),data_sha256={},sample_ids={},correct_counts={},sources={});arrays={}
        for split,values in [('selection_dev',dev),('selection_holdout',hold)]:
            arrays[split]=np.stack([values[t] for t in tags]);metadata['data_sha256'][split]=sha(root/'data/mnli'/f'{split}.jsonl');metadata['sample_ids'][split]=[r['sample_id'] for r in data_rows];metadata['correct_counts'][split]=list(map(int,arrays[split].sum(1)));metadata['sources'][split]={}
        bundle=root/'results/frozen_correctness.npz';np.savez_compressed(bundle,tags=np.array(tags),metadata=np.array(json.dumps(metadata)),**arrays)
        subprocess.run([sys.executable,str(CODE_ROOT/'analyze_selection.py'),'--analysis-spec',str(specpath),'--prediction-bundle',str(bundle)],env=dict(os.environ,BENCH_EXP_DIR=str(root),BENCH_SOURCE_DIR=str(CODE_ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        r=json.loads((root/'results/final_selection_analysis.json').read_text());cv=next(x for x in r['groups'] if x['method']=='cv')
        expected=dict(direct_deploy_accuracy=10/12,selection_oracle_accuracy=6/12,oracle_dev_selection_accuracy=10/12,selection_delta_vs_oracle_dev_choice=-4/12,selection_delta_vs_random=-2/12,selection_delta_vs_constant_initial_choice=-4/12,regret_to_best_oracle_holdout_ceiling=4/12,train_all_original_pool_accuracy=11/12,selection_delta_vs_train_all_original_pool=-5/12)
        for field,value in expected.items():assert abs(cv[field]['mean']-value)<1e-12,(field,cv[field],value)
        assert all(x['selected_subset']=='candidate_4' for x in r['per_start'] if x['method']=='cv')
        assert r['direct_paired_comparisons'][0]['delta']['mean']==0,'Equal generated holdout models must have zero paired quality gap despite different dev selection'
        full=next(x for x in r['direct_quality'] if x['family']=='fullpool');assert full['delta_vs_oracle_same_candidate']['mean']==0,'Fullpool must use its own reference'
        write(root/'results/frozen_correctness_audit.json',dict(passed=True,sha256=sha(bundle),freeze_sha256=sha(freeze),decision_sha256=sha(decisionpath)))
        (root/'scripts').symlink_to(CODE_ROOT,target_is_directory=True)
        public_command=[sys.executable,str(CODE_ROOT/'reproduce_surrogate_statistics.py'),'--study-dir',str(root),'--report-dir',str(root),'--data-dir',str(root/'data/mnli'),'--output',str(root/'public_replay')]
        subprocess.run(public_command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        public=json.loads((root/'public_replay/reproduction_audit.json').read_text());assert public['passed'] and all(public['checked_fields'].values())

    record=dict(passed=True,expected_means=expected,checks=['Dev-selected bad candidate remains fixed on holdout','Selected Oracle and direct generated model outcomes are distinct','Random, constant and train-all baselines use correct candidates','Fullpool uses its population-specific reference','Paired model quality can be equal while candidate-selection outcomes differ'],scope='Synthetic CPU integration audit only; no experimental selection outcomes accessed and no neural performance claim.',analysis_sha256=sha(CODE_ROOT/'analyze_selection.py'),public_cpu_reproduction_passed=True,public_reproduction_sha256=sha(CODE_ROOT/'reproduce_surrogate_statistics.py'))
    write(EXP/'results/selection_analysis_cpu_audit.json',record);log(record)

if __name__=='__main__':main()
