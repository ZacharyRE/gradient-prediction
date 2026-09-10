"""Verify that exact-question subsets retain the intended parent targets unchanged."""
from common import *
records=[]
mapping={'sample_common':'sample_all','teacher_common':'teacher_all',
         'teacher32_common_teacher7':'teacher32_all','teacher7_common_teacher32':'teacher_all'}
loaded={}
for subset,parent in mapping.items():
    sp=S/f'data/{subset}.jsonl';pp=S/f'data/{parent}.jsonl'
    rows=read(sp);parents=read(pp);lookup={r['original_index']:r for r in parents}
    assert len(lookup)==len(parents) and len({r['original_index'] for r in rows})==len(rows)
    assert all(r==lookup[r['original_index']] for r in rows)
    loaded[subset]=rows
    records.append(dict(subset=subset,parent=parent,n=len(rows),all_full_rows_equal_parent=True,
        subset_sha256=sha(sp),parent_sha256=sha(pp),stored_target_tokens=sum(r['target_tokens'] for r in rows),
        target_sources=sorted({r['target_source'] for r in rows})))
pairs=[]
for first,second in [('sample_common','teacher_common'),('teacher32_common_teacher7','teacher7_common_teacher32')]:
    a,b=loaded[first],loaded[second]
    assert [(r['original_index'],r['problem']) for r in a]==[(r['original_index'],r['problem']) for r in b]
    pairs.append(dict(first=first,second=second,n=len(a),same_question_order=True,
                      identical_target_texts=sum(x['solution']==y['solution'] for x,y in zip(a,b))))
write(S/'audits/matched_generator_data_contract.json',dict(time=time.time(),passed=True,
    script_sha256=sha(__file__),subsets=records,pairs=pairs,
    scope='Data provenance and exact question matching, not certification of mathematical process correctness. '
          'Stored target-token counts retain the original builder convention.'))
print(json.dumps(dict(subsets=records,pairs=pairs),indent=2))
