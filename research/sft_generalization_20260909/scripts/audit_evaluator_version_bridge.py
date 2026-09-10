"""Verify the archived dispatch-only source change and both actual 500-text replays."""
import ast
from common import *
change_path=S/'audits/evaluator_dispatch_change.json'
replay_path=S/'audits/dispatch_v2_protocol_replay.json'
change=json.loads(change_path.read_text());replay=json.loads(replay_path.read_text())
old=S/'audits/evaluate_dispatch_v1.py';new=S/'audits/evaluate_dispatch_v2.py'
assert sha(old)==change['old_script_sha256']==replay['old_source_sha256']
assert sha(new)==change['new_script_sha256']==replay['new_source_sha256']==sha(S/'scripts/evaluate.py')
def without_queue(path):
    tree=ast.parse(path.read_text());found=0
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse('a.queue',mode='eval').body):
            node.body=[ast.Pass()];found+=1
    assert found==1
    return ast.dump(tree,include_attributes=False)
assert without_queue(old)==without_queue(new)
root=S/'results/evaluation/dev';records=[]
for entry in replay['checks']:
    rp=root/entry['reference']/'math_predictions.jsonl';np=root/entry['replay']/'math_predictions.jsonl'
    assert sha(rp)==entry['reference_sha256'] and sha(np)==entry['replay_sha256']
    rr,nr=read(rp),read(np);assert len(rr)==len(nr)==entry['n']==500
    fields=['sample_hash','prediction','correct','finish_reason','generated_tokens']
    counts={k:sum(a[k]==b[k] for a,b in zip(rr,nr)) for k in fields}
    assert all(v==500 for v in counts.values())
    rm=json.loads(rp.with_name('math_manifest.json').read_text())
    nm=json.loads(np.with_name('math_manifest.json').read_text())
    assert rm['script_sha256']==sha(old) and nm['script_sha256']==sha(new)
    for key in ['model','adapter','data_sha256','engine','batch_invariant','max_tokens','temperature','chunk_size']:
        assert rm[key]==nm[key],(entry['reference'],key)
    records.append(dict(reference=entry['reference'],replay=entry['replay'],equal_counts=counts,
                        reference_manifest_sha256=sha(rp.with_name('math_manifest.json')),
                        replay_manifest_sha256=sha(np.with_name('math_manifest.json')),
                        reference_predictions_sha256=sha(rp),replay_predictions_sha256=sha(np)))
assert len(records)==2 and {x['reference'] for x in records}=={'base','previous_self_sdpa_s43'}
write(S/'audits/evaluator_version_bridge.json',dict(time=time.time(),passed=True,script_sha256=sha(__file__),
    allowed_development_script_sha256=[sha(old),sha(new)],current_script_sha256=sha(new),
    source_change_audit_sha256=sha(change_path),original_replay_audit_sha256=sha(replay_path),
    ast_equal_after_removing_only_queue_dispatch=True,comparisons=records,
    scope='Two explicit archived evaluator versions, differing only in queue dispatch. Baseline and nonzero '
          'adapter reproduce all500 texts/flags/lengths. This is not permission for arbitrary script changes '
          'or a proof of every inference backend equivalence. Final new benchmark outputs use current version only.'))
print('Two archived evaluator versions verified: only queue dispatch differs; 1000 full outputs replay exactly.')
