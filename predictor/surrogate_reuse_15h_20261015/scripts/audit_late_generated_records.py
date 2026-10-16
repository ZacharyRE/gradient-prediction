"""Reconcile post-freeze development generations with their full source inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
from evaluate import grade,GRADING_REVISION

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True)
    rows=read(data('mnli','accuracy_dev'));prompts=[prompt(tok,r) for r in rows]
    lengths=[len(tok.encode(p,add_special_tokens=False)) for p in prompts]
    entries=[];protocols={};tick=time.time()
    for path in sorted((EXP/'results/eval').glob('*/summary.json')):
        name=path.parent.name
        if not (name.startswith(('late_diversity_','late_capacity_','late_kd','optional_completed_'))
                or name in ['pilot_table','pilot_token','pilot_needed','pilot_class_batched']):continue
        protocol=json.loads((path.parent/'protocol.json').read_text())
        assert protocol['args']['split']=='accuracy_dev'
        assert protocol['input_sha256']['mnli']==sha(data('mnli','accuracy_dev'))
        assert protocol['full_inputs'] and protocol['input_token_ids_verified_at_generation']
        assert protocol['max_prompt_tokens']==max(lengths) and protocol['max_new_tokens']==256
        assert max(lengths)+256<=protocol['context']
        protocols[name]=sha(path.parent/'protocol.json')
        for key,summary in json.loads(path.read_text()).items():
            records_path=path.parent/f'{key}.jsonl';records=read(records_path)
            assert len(records)==len(rows)==summary['n']
            counts=dict(correct=0,strict_correct=0,legacy_correct=0,output_limit_hits=0)
            for actual,source,p in zip(records,rows,prompts):
                assert actual['sample_id']==source['sample_id'] and actual['gold']==source['gold']
                assert actual['problem']==source['problem'] and actual['full_prompt']==p
                assert all(actual[k]==v for k,v in grade(actual['prediction'],source).items())
                assert actual['tokens']<=256
                for k in ['correct','strict_correct','legacy_correct']:counts[k]+=int(actual[k])
                counts['output_limit_hits']+=actual['finish_reason']=='length'
            assert counts['correct']==summary['correct'] and counts['legacy_correct']==summary['legacy_correct']
            assert counts['output_limit_hits']==summary['limit_hits']
            assert abs(counts['strict_correct']/len(rows)-summary['strict_accuracy'])<1e-12
            entries.append(dict(evaluation=name,tag=key,records_sha256=sha(records_path),**counts))
    assert entries
    audited={(r['evaluation'],r['tag']):r for r in entries}
    late=json.loads((EXP/'results/late_development_summary.json').read_text())
    for row in late['rows']:
        checked=audited[row['evaluation'],row['tag']+'__mnli']
        assert abs(row['accuracy']-100*checked['correct']/len(rows))<1e-10
    result=dict(passed=True,epoch=time.time(),seconds=time.time()-tick,
                source_data_sha256=sha(data('mnli','accuracy_dev')),examples_per_entry=len(rows),
                records=len(rows)*len(entries),entries=entries,protocol_sha256=protocols,
                maximum_full_input_tokens=max(lengths),maximum_new_output_tokens=256,
                late_summary_sha256=sha(EXP/'results/late_development_summary.json'),
                late_summary_rows_verified=len(late['rows']),
                grading_revision=GRADING_REVISION,
                scope='Every completed late development record: complete source problem/chat prompt, actual-generation input-ID protocol, and recomputed stored grading. Same frozen grader, not independent semantic regrading. Duplicate evaluations remain separately counted. Partial or unexecuted recipes are not inferred to have outcomes.')
    write(EXP/'results/late_generated_records_audit.json',result)
    log(dict(passed=True,entries=len(entries),records=result['records']))

if __name__=='__main__':main()
