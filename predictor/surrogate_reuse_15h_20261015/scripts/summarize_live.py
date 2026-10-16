"""Development-only working tables; never select from test results."""
from common import *
import csv

def main():
    evaluations={}
    for path in (EXP/'results/eval').glob('*/summary.json'):
        protocol=json.loads((path.parent/'protocol.json').read_text())
        if protocol['args']['split']!='accuracy_dev':continue
        for key,value in json.loads(path.read_text()).items():evaluations[key]=value
    rows=[]
    for path in (EXP/'models').glob('*/summary.json'):
        summary=json.loads(path.read_text())
        if 'entries' not in summary:continue
        cfg=json.loads((path.parent/'config.json').read_text());job=cfg['job']
        if 'smoke' in job['name']:continue
        hist=json.loads((path.parent/'history.json').read_text())
        for e in summary['entries']:
            v=evaluations.get(e['tag']+'__'+e['task'],{})
            seen=[r for r in hist if r['step']<=e['steps']]
            last=next(r for r in hist if r['step']==e['steps'])
            row=dict(task=e['task'],method=e['method'],seed=e['seed'],predictor_seed=e['predictor_seed'],steps=e['steps'],lr=e['lr'],dev_n=v.get('n',''),dev_accuracy_pct=100*v['accuracy'] if v else '',dev_ce128=e['dev_ce'],online_seconds=e['online_seconds'],prefix_preparation_seconds=e.get('cache_seconds',0),offline_fit_seconds=e.get('offline_fit_seconds',0),online_true_examples=sum(r['true_gradient_examples'] for r in seen),calibration_seconds=sum(r.get('calibration_seconds',0) for r in seen),fast_loss=job.get('fast_loss',job['kind']=='online_ridge'),kind=job['kind'],tag=e['tag'],source_config_sha256=sha(path.parent/'config.json'))
            audit=last.get('gradient_audit',{})
            for key in ['cosine','relative_l2','norm_ratio']:row['batch_gradient_'+key]=audit.get('batch_lora',{}).get(key,'')
            row['cv_residual_variance_ratio']=audit.get('cv_residual_variance_ratio','')
            rows.append(row)
    rows.sort(key=lambda r:(r['task'],r['method'],r['seed'],r['steps']))
    out=EXP/'results/development_live.csv'
    if rows:
        with out.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    log(dict(completed_trajectories=len({r['tag'].rsplit('__step',1)[0] for r in rows}),checkpoint_rows=len(rows),accuracy_readouts=sum(r['dev_n']!='' for r in rows)))
    for task in sorted({r['task'] for r in rows}):
        final=[r for r in rows if r['task']==task and r['dev_n']!='']
        by_method={}
        for r in final:
            key=(r['method'],r['seed']);old=by_method.get(key)
            if old is None or (r['dev_accuracy_pct'],-r['online_seconds'])>(old['dev_accuracy_pct'],-old['online_seconds']):by_method[key]=r
        for r in sorted(by_method.values(),key=lambda x:(-x['dev_accuracy_pct'],x['online_seconds'])):
            log({k:r[k] for k in ['task','method','seed','steps','dev_accuracy_pct','online_seconds','prefix_preparation_seconds']})

if __name__=='__main__':main()
