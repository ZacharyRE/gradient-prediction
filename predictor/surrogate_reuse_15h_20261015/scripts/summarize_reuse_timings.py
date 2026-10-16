"""Summarize exact-weight standalone replays; never use profiled times as speed."""
from common import *
from collections import defaultdict
import csv

def main():
    groups=defaultdict(lambda:dict(timing=[],flops=[]))
    for folder,kind in [('reuse_timing','timing'),('reuse_flops','flops')]:
        for path in sorted((EXP/'results'/folder).glob('*.json')):
            r=json.loads(path.read_text());e=r['reference_entry']
            key=(e['tag'],r['reference_sha256'],r.get('actual_teacher_backend','native'))
            groups[key][kind].append((path,r))
    rows=[];detail=[]
    for (tag,weight_sha,backend),g in sorted(groups.items()):
        valid=[(p,r) for p,r in g['timing'] if r['weights_bitwise_equal'] and r['timing_valid_for_reference']]
        # A repeated invocation with the same repeat id is a retry, not another
        # independent registered repetition. Preserve all attempts in receipts.
        repeats={}
        for p,r in valid:repeats.setdefault(r['repeat'],[]).append((p,r))
        chosen=[sorted(v,key=lambda x:x[1]['started_epoch'])[-1] for _,v in sorted(repeats.items())]
        profile=[(p,r) for p,r in g['flops'] if r['weights_bitwise_equal']]
        e=(g['timing'] or g['flops'])[0][1]['reference_entry']
        row=dict(tag=tag,method=e['method'],kind=e['kind'],warmup_seed=e['warmup_seed'],subset=e['subset'],steps=e['steps'],lr=e['lr'],teacher_backend=backend,qualified_repetitions=len(chosen),three_repeats_complete=len(chosen)>=3,weight_sha256=weight_sha)
        for name in ['online_seconds','preparation_seconds','model_setup_seconds','data_encoding_seconds','pipeline_seconds','script_seconds']:
            values=[r[name] for _,r in chosen if r.get(name) is not None]
            for suffix,fn in [('median',np.median),('min',min),('max',max)]:row[name+'_'+suffix]=float(fn(values)) if values else ''
        values=[r['preparation_seconds']+r['online_seconds'] for _,r in chosen]
        row['preparation_plus_online_seconds_median']=float(np.median(values)) if values else ''
        totals=[r['total_matrix_flops'] for _,r in profile]
        if totals:assert len(set(totals))==1,(tag,'Repeated FLOP profiles disagree')
        row['total_matrix_flops']=totals[0] if totals else ''
        rows.append(row)
        detail.append(dict(**row,selected_timing_receipts=[dict(path=str(p),sha256=sha(p)) for p,_ in chosen],all_attempts=[dict(path=str(p),sha256=sha(p),weights_bitwise_equal=r['weights_bitwise_equal']) for k in ['timing','flops'] for p,r in g[k]]))
    write(EXP/'results/standalone_reuse_timing_summary.json',dict(rows=detail,scope='Exact-weight fresh-process fixed-stop replays. Profiled timings excluded. At least3 distinct repeat IDs required for final timing qualification. Source/target warmups, offline predictor fitting and evaluation excluded; charge separately. Script time ends before post-run weight comparison. Matrix FLOPs omit scalar/nonlinear operations.'))
    if rows:
        with (EXP/'results/standalone_reuse_timing_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    log(dict(reference_adapters=len(rows),qualified_three_repeats=sum(r['three_repeats_complete'] for r in rows)))

if __name__=='__main__':main()
