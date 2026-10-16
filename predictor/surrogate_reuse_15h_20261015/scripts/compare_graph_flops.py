"""Audit opaque-replay FLOPs against the independently executed eager recipe."""
from common import *

records={}
for path in (EXP/'results/reuse_flops').glob('*.json'):
    r=json.loads(path.read_text());tag=r['reference_entry']['tag']
    for mode in ['graph','eager_static']:
        if tag==f'mnli__w301_candidate_0_graph_cv4_u401_graph_smoke_{mode}__step24':records[mode]=(path,r)
assert set(records)=={'graph','eager_static'}
g,e=[records[m][1] for m in ['graph','eager_static']]
assert g['weights_bitwise_equal'] and e['weights_bitwise_equal']
assert g['actual_sha256']==e['actual_sha256']
assert g['preparation_phase_flops']==e['preparation_phase_flops']
receipt=g['teacher_backend_flop_accounting']
extra=sum(r['matrix_flops_per_call']*(r['capture_warmup_backward_examples']+r['audit_backward_examples'])//r['batch_size'] for r in receipt['shapes'].values())
observed=g['total_matrix_flops']-e['total_matrix_flops']
assert observed==extra,(observed,extra)
report=dict(passed=True,graph_total_matrix_flops=g['total_matrix_flops'],eager_total_matrix_flops=e['total_matrix_flops'],extra_graph_warmup_and_audit_matrix_flops=extra,observed_difference=observed,weights_bitwise_equal=True,prefix_preparation_identical=True,sources={m:dict(path=str(p),sha256=sha(p)) for m,(p,r) in records.items()},scope='24-step same-padding CV4. Graph adjusted FLOPs equal separately profiled eager execution plus actually executed capture warmups and audits; tracing is excluded. Matrix/convolution arithmetic only; not latency or all scalar operations.')
write(EXP/'results/graph_flop_accounting_audit.json',report);log(report)
