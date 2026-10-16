"""Lock final scientific recipes without consulting selection outcomes."""
from common import *

def main():
    assert not (EXP/'configs/surrogate_final_freeze.json').exists()
    for p in (EXP/'results/eval').glob('*/protocol.json'):
        assert json.loads(p.read_text())['args']['split'] not in ['selection_dev','selection_holdout']
    starts=[302,303,304];subsets=[f'candidate_{i}' for i in range(5)]+['genre_'+x for x in ['fiction','government','slate','telephone','travel']]
    groups=[]
    def add(name,suffix,sets,selectors):
        sources=[str(EXP/'results'/f'entries_mnli_w{w}_{s}_u{u}{suffix}.json') for w,u in zip(starts,[401,402,403]) for s in sets]
        groups.append(dict(name=name,sources=sources,selectors=selectors,expected_entries=len(sources)*len(selectors)))
    def sel(method,steps):return dict(method=method,steps=steps)
    add('candidate_matched_and_short_oracle','_blindmatched',subsets,[sel(m,s) for m in ['oracle32_matched','oracle16_matched','cv16_matched'] for s in [128,256,512]]+[sel('oracle32_matched',64)])
    add('candidate_independently_tuned_oracles','_blindtunedcandidate',subsets,[sel('oracle32_lr0.0003',128),sel('subset16_lr0.0003',256)])
    add('candidate_zero_gradient_control','_blindzerocandidate',subsets,[sel('zero_every4_lr0.0003',256)])
    add('fullpool_matched_and_short_oracle','_blindfullpool',['update'],[sel(m,s) for m in ['oracle32_matched','oracle16_matched','cv16_matched'] for s in [256,512]]+[sel('oracle32_matched',128)])
    add('fullpool_tuned_oracle_and_zero_control','_blindfullpoolcontrol',['update'],[sel('subset16_lr0.0003',512),sel('zero_every4_lr0.001',256)])
    add('fullpool_independently_tuned_original_cv16','_fullpooltunedcv',['update'],[sel('cv16_lr0.0003',512)])
    add('fullpool_graph_cv8_and_matched_oracle8','_fullpoolcv8graph',['update'],[sel('graph_cv8_lr0.001',512),sel('graph_subset8_lr0.001',512)])
    add('fullpool_state_predictor_early_cutoff','_fullpoolstate',['update'],[sel('state256_cv16_until32_lr0.001',256)])
    plan=dict(complete=True,registered_at=time.time(),heldout_warmups=starts,expected_unique_entries=sum(g['expected_entries'] for g in groups)+3,entry_groups=groups,behavior_models=[str(EXP/'models/mnli_behavior_anchored128_coverage/best.pt')],selection_basis='Development only. Original CV16 independently tuned by population; graph CV8 fullpool512 and state256 early-cutoff fullpool256 selected after three-start confirmation. Template and original early-cutoff ablations do not replace selected state model. No further method families will be promoted. Candidate0 was used for development; other random subsets are new combinations, genre groups are source-disjoint.',primary_comparators='Independently tuned Oracle32, short Oracle32, matched Oracle16/8, zero-gradient Adam control and warmup; matched and practical comparisons reported separately.',timing_plan=dict(repetitions=3,starts=starts,candidate_subset='candidate_0',candidate_recipes=[sel('oracle32_matched',64),sel('oracle32_lr0.0003',128),sel('oracle16_matched',128),sel('cv16_matched',128),sel('subset16_lr0.0003',256),sel('zero_every4_lr0.0003',256)],fullpool_recipes=[sel('oracle32_matched',256),sel('oracle16_matched',256),sel('cv16_matched',256),sel('subset16_lr0.0003',512),sel('cv16_lr0.0003',512),sel('zero_every4_lr0.001',256),sel('graph_cv8_lr0.001',512),sel('graph_subset8_lr0.001',512),sel('state256_cv16_until32_lr0.001',256)],flops_starts=[302],qualification='Three interleaved fresh-process repeats per selected adapter, bitwise weight equality required. FLOPs profiled separately on first heldout start; all candidate subsets retain measured training component costs. No timing-based recipe reselection.'))
    assert plan['expected_unique_entries']==432
    path=EXP/'configs/surrogate_final_recipe_plan.json'
    assert not path.exists()
    write(path,plan);log(dict(entries=432,groups=len(groups),plan_sha256=sha(path)))

if __name__=='__main__':main()
