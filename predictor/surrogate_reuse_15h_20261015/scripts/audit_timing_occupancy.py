"""Conservatively flag stage timings using sampled GPU occupancy, without filtering accuracy."""
from common import *

def main():
    holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
    known={holder['pid']}|{int(p.stem) for p in (EXP/'results/worker_memory_reservations').glob('*.json')}
    observations=read(EXP/'results/gpu_observations.jsonl')
    rows=[]
    for path in sorted((EXP/'results').glob('stage_*.json')):
        stage=json.loads(path.read_text())
        if 'started' not in stage or 'finished' not in stage:continue
        samples=[o for o in observations if stage['started']-15<=o['time']<=stage['finished']+15]
        foreign=[]
        for sample in samples:
            pids=[int(line.split(',')[1]) for line in sample['processes']]
            if any(pid not in known for pid in pids):foreign.append(sample['time'])
        rows.append(dict(stage=stage['name'],started=stage['started'],finished=stage['finished'],returncode=stage['returncode'],sample_count=len(samples),foreign_context_sample_count=len(foreign),timing_status='requires_clean_replay' if foreign or stage.get('timing_requires_clean_replay') else 'sampled_owned_only' if samples else 'unverified'))
    result=dict(updated=time.time(),stages=rows,scope='Conservative15-second margins around stage intervals. Sampling cannot exclude activity between observations. Foreign CUDA contexts need not imply active computation; affected timings are excluded from final speed claims regardless. Numerical accuracy results remain eligible.')
    write(EXP/'results/timing_occupancy_audit.json',result)
    log(dict(stages=len(rows),requires_clean_replay=sum(r['timing_status']=='requires_clean_replay' for r in rows)))

if __name__=='__main__':main()
