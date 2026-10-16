"""Reconcile full saved evaluation inputs and grading with frozen source rows."""
from common import *
from evaluate import grade,GRADING_REVISION
from surrogate_freeze import FREEZE,DECISIONS

def main():
    # This audit is CPU only and must not create a CUDA context beside a live stage.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from transformers import AutoTokenizer
    frozen=json.loads(FREEZE.read_text());tok=AutoTokenizer.from_pretrained(MODELS[frozen['model']]);results={};tick=time.time()
    for split in ['selection_dev','selection_holdout']:
        root=EXP/'results/eval'/f'surrogate_final_{split}';protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text());rows=read(data('mnli',split));prompts=[prompt(tok,r) for r in rows];lengths=[len(tok.encode(s,add_special_tokens=False)) for s in prompts]
        assert set(summary)==set(frozen['entries']) and max(lengths)==protocol['max_prompt_tokens']
        assert max(lengths)+protocol['max_new_tokens']<=protocol['context']
        entries=[]
        for key in sorted(frozen['entries']):
            path=root/f'{key}.jsonl';rr=read(path);assert len(rr)==len(rows)
            counts=dict(correct=0,strict_correct=0,has_box=0,literal_option_text=0,legacy_correct=0,output_limit_hits=0,valid_abc_answer=0)
            for actual,source,expected_prompt in zip(rr,rows,prompts):
                assert actual['sample_id']==source['sample_id'] and actual['gold']==source['gold']
                assert actual['problem']==source['problem'] and actual['full_prompt']==expected_prompt,'Saved evaluation input differs from full source prompt'
                graded=grade(actual['prediction'],source);assert all(actual[k]==v for k,v in graded.items()),'Saved grading differs from frozen grader'
                assert actual['tokens']<=protocol['max_new_tokens']
                for name in ['correct','strict_correct','has_box','literal_option_text','legacy_correct']:counts[name]+=int(actual[name])
                counts['output_limit_hits']+=actual['finish_reason']=='length';counts['valid_abc_answer']+=actual['extracted_answer'] in ['A','B','C']
            assert counts['correct']==summary[key]['correct'] and counts['legacy_correct']==summary[key]['legacy_correct'] and counts['output_limit_hits']==summary[key]['limit_hits']
            assert abs(counts['strict_correct']/len(rows)-summary[key]['strict_accuracy'])<1e-12
            entries.append(dict(tag=key,n=len(rows),**counts,record_sha256=sha(path)))
        results[split]=dict(entries=entries,examples_per_entry=len(rows),max_full_input_tokens=max(lengths),output_generation_cap=protocol['max_new_tokens'],context_capacity=protocol['context'],total_records=len(rows)*len(entries),complete_source_prompts_verified=True,grading_fields_recomputed=True,protocol_sha256=sha(root/'protocol.json'))
    result=dict(passed=True,seconds=time.time()-tick,freeze_sha256=sha(FREEZE),decisions_sha256=sha(DECISIONS),grading_revision=GRADING_REVISION,splits=results,scope='Every saved problem and chat prompt is compared with the complete frozen source row and recorded base-model tokenizer. All stored grading fields are recomputed using the frozen grader; this validates record/summary consistency, not an independent semantic grading policy. Actual generation additionally verified input token IDs. The 256-token cap limits outputs, not inputs.')
    write(EXP/'results/generated_records_audit.json',result);log(dict(passed=True,seconds=result['seconds'],records={k:v['total_records'] for k,v in results.items()}))

if __name__=='__main__':main()
