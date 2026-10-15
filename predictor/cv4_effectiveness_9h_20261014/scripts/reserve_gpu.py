"""Finite-duration GPU reservation, compatible with this study's worker cache."""
from common import *
import fcntl, signal

def main():
    cfg=json.loads((EXP/'configs/protocol.json').read_text())
    lock=open('/tmp/h200_gpu_admission_0.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.startswith(cfg['gpu_uuid']) for line in output.splitlines()),'GPU no longer vacant'
    import torch
    torch.cuda.set_device(0)
    stopping=False
    def stop(*args):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    blocks=[];gib=1024**3;last_mode=None
    state=dict(pid=os.getpid(),process_start=process_start(os.getpid()),gpu_index=0,gpu_uuid=cfg['gpu_uuid'],started=time.time(),cutoff_epoch=cfg['deadline_epoch'])
    try:
        while not stopping and time.time()<cfg['deadline_epoch']:
            req=json.loads((EXP/'results/gpu_reservation_request.json').read_text())
            mode=req['mode'];assert mode in ['idle','stage']
            free,total=torch.cuda.mem_get_info();target=int(total/gib)-(6 if mode=='idle' else 52)
            changed=False
            while len(blocks)>target:blocks.pop();changed=True
            if changed:torch.cuda.empty_cache()
            if mode=='idle' or last_mode!=mode:
                free,_=torch.cuda.mem_get_info()
                for _ in range(min(max(0,target-len(blocks)),max(0,int(free/gib)-6))):
                    try:blocks.append(torch.empty(gib,dtype=torch.uint8,device='cuda'))
                    except torch.OutOfMemoryError:torch.cuda.empty_cache();break
            last_mode=mode
            write(EXP/'results/gpu_reservation.json',dict(**state,status='ready',mode=mode,token=req['token'],updated=time.time(),reserved_gib=len(blocks)))
            time.sleep(.5)
    finally:
        blocks.clear();torch.cuda.empty_cache()
        write(EXP/'results/gpu_reservation.json',dict(**state,status='stopped',updated=time.time(),reserved_gib=0))

if __name__=='__main__':main()
