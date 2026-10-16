"""Keep post-freeze exploratory readouts separate from final confirmation."""
from common import *
import csv

def main():
    from summarize_reuse import rows
    selected=[r for r in rows if r['evaluation'].startswith(('late_diversity_', 'late_capacity_', 'late_kd','optional_completed_'))
              or r['evaluation'] in ['pilot_table','pilot_token','pilot_needed','pilot_class_batched']]
    by_tag={}
    for r in selected:
        if r['tag'] in by_tag:assert by_tag[r['tag']]['accuracy']==r['accuracy']
        else:by_tag[r['tag']]=r
    selected=list(by_tag.values())
    groups=[]
    for method,steps in sorted({(r['method'],r['steps']) for r in selected}):
        rr=[r for r in selected if (r['method'],r['steps'])==(method,steps)]
        assert len({(r['warmup_seed'],r['seed']) for r in rr})==len(rr)
        groups.append(dict(method=method,steps=steps,starts=[r['warmup_seed'] for r in rr],
                           mean_accuracy_pct=float(np.mean([r['accuracy'] for r in rr])),
                           mean_preparation_updates_seconds=float(np.mean([r['adaptation_seconds'] for r in rr])),
                           output_cap_hits=sum(r['output_cap_hits'] for r in rr)))
    diversity=[r for r in selected if r['method'].startswith('diversity')]
    paired=[]
    for early in [False,True]:
        suffix='cv16_until32_lr0.001' if early else 'cv16_lr0.001'
        for steps in [32,128,256,512]:
            by={n:{r['warmup_seed']:r for r in diversity if r['method']==f'{n}_{suffix}' and r['steps']==steps}
                for n in ['diversity2048','diversity512_control']}
            common=sorted(set(by['diversity2048'])&set(by['diversity512_control']))
            if not common:continue
            delta=[by['diversity2048'][w]['accuracy']-by['diversity512_control'][w]['accuracy'] for w in common]
            paired.append(dict(early_cutoff=early,steps=steps,starts=common,complete=common==[301,311,321],
                               differences_pp=delta,mean_difference_pp=float(np.mean(delta))))
    fitting={}
    for name in ['state256_coverage_fast','state256_diversity2048','state512_diversity2048','token_kd128','token_kd128_nograd','token_kd256']:
        root=EXP/'models'/f'mnli_critic_{name}'
        if not (root/'summary.json').exists():continue
        summary=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
        history=json.loads((root/'history.json').read_text())
        best=next(r for r in history if r['epoch']==summary['best_epoch'])
        fitting[name]=dict(summary=summary,config=cfg,selected_development_diagnostics=best,
                          summary_sha256=sha(root/'summary.json'))
    diversity_fit_audit=None
    if 'state256_diversity2048' in fitting:
        old=fitting['state256_coverage_fast']['config'];new=fitting['state256_diversity2048']['config']
        assert old['states']==new['states'] and len(new['states'])==12
        assert old['train_examples']*old['job']['epochs']==new['train_examples']*new['job']['epochs']==245760
        cache=EXP/'cache/critic_diversity2048'
        for name in new['states']:
            original=EXP/'cache'/('critic_fast_v1' if '_fast_' in name else 'critic_v1')/name
            a=json.loads((original/'predictor_train/manifest.json').read_text())
            b=json.loads((cache/name/'predictor_train/manifest.json').read_text())
            assert len(a['indices'])==512 and len(b['indices'])==2048
            assert a['indices']==b['indices'][:512] and a['data_sha256']==b['data_sha256']
            da=json.loads((original/'predictor_dev/manifest.json').read_text())
            db=json.loads((cache/name/'predictor_dev/manifest.json').read_text())
            assert da['indices']==db['indices'] and da['data_sha256']==db['data_sha256']
        diversity_fit_audit=dict(passed=True,states=12,old_unique_inputs=512,new_unique_inputs=2048,
                                 unchanged_predictor_dev=True,nested_input_subset=True,
                                 example_state_visits_each=245760,
                                 optimizer_steps={n:fitting[n]['summary']['optimizer_steps'] for n in ['state256_coverage_fast','state256_diversity2048']})
    result=dict(epoch=time.time(),scope='Post-freeze exploratory development only; no final-method replacement. Historical component times, not standalone repetitions. Three-start comparisons use paired301/401,311/411,321/421; other screens are single-start.',
                rows=selected,groups=groups,diversity_paired=paired,fitting=fitting,diversity_fit_audit=diversity_fit_audit)
    write(EXP/'results/late_development_summary.json',result)
    if groups:
        with (EXP/'results/late_development_summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0]),lineterminator='\n');writer.writeheader();writer.writerows(groups)
    log(dict(readouts=len(selected),groups=len(groups),paired_diversity_complete=bool(paired) and all(r['complete'] for r in paired)))

if __name__=='__main__':main()
