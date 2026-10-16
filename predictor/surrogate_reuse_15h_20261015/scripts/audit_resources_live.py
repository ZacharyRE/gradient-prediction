"""Reconcile sampled GPU processes with owned reservation/worker receipts."""
from common import *
import collections

def main():
    config=json.loads((EXP/'configs/protocol.json').read_text())
    holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
    known={holder['pid']}|{int(p.stem) for p in (EXP/'results/worker_memory_reservations').glob('*.json')}
    history_path=EXP/'provenance/gpu_holder_history.json'
    history=json.loads(history_path.read_text())['holders'] if history_path.exists() else []
    observations=read(EXP/'results/gpu_observations.jsonl');pids=collections.Counter();unmatched=[]
    for row in observations:
        for line in row['processes']:
            uuid,pid,memory=[x.strip() for x in line.split(',')];pid=int(pid)
            assert uuid==config['gpu_uuid'];pids[pid]+=1
            historical_holder=any(r['pid']==pid and r['first_possible_epoch']<=row['time']<=r.get('last_possible_epoch',time.time()) for r in history)
            if pid not in known and not historical_holder:unmatched.append(dict(time=row['time'],pid=pid,memory_mib=memory))
    result=dict(updated=time.time(),gpu_index=config['gpu_index'],gpu_uuid=config['gpu_uuid'],observation_count=len(observations),observed_processes=dict(pids),known_owned_process_count=len(known),unmatched_process_observations=unmatched,observed_only_receipted_owned_processes=not unmatched,elapsed_seconds=time.time()-config['start_epoch'],authorized_seconds=config['deadline_epoch']-config['start_epoch'],scope='One GPU, cooperative admission lock plus held VRAM and reusable worker allocator cache; sampled every15seconds. Historical holder renewals use verified transition time ranges; worker matching uses recorded PIDs and is not continuous process-lifetime attestation. This is not administrator-enforced exclusive mode and cannot rule out activity between samples. Reservation memory is not algorithm memory usage.')
    write(EXP/'results/resource_observation_audit.json',result)
    log({k:v for k,v in result.items() if k not in ['observed_processes','scope']})

if __name__=='__main__':main()
