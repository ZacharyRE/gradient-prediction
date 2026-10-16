"""Resume unchanged optional queues only after all primary GPU validation finishes."""
from common import *
import sys

def main():
    prerequisite=EXP/'results/selection_pipeline_summary.json';registration=json.loads((EXP/'provenance/research_queue_deferral_registration.json').read_text())
    while True:
        if prerequisite.exists() and json.loads(prerequisite.read_text()).get('complete'):break
        if deadline(5400):
            write(EXP/'results/optional_research_resumption.json',dict(resumed=False,epoch=time.time(),reason='Final validation consumed the optional research reserve; unexecuted recipes remain unevaluated.'));return
        time.sleep(30)
    children=[]
    for queue in sorted({r['queue'] for r in registration['deferred']}):
        if deadline(5400):break
        name=Path(queue).stem+'_after_final_validation';log_path=EXP/'logs'/f'{name}.log'
        with log_path.open('a') as out:
            child=subprocess.Popen([sys.executable,str(EXP/'scripts/run_queue.py'),'--queue',queue],env=dict(os.environ,BENCH_RUN_ID=EXP.name),stdout=out,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        children.append(dict(queue=queue,pid=child.pid,process_start=process_start(child.pid),log=str(log_path)))
    write(EXP/'results/optional_research_resumption.json',dict(resumed=bool(children),epoch=time.time(),children=children,scope='Same predeclared development queues; no changes to frozen final recipes or selection decisions.'))

if __name__=='__main__':main()
