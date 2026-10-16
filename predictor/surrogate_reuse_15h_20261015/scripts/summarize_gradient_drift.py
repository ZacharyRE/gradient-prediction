"""Export descriptive gradient diagnostics; never select checkpoints from them."""
from common import *
import csv,collections

def main():
    rows=[]
    for path in sorted((EXP/'models').glob('*/history.json')):
        config_path=path.parent/'config.json'
        if not config_path.exists():continue
        cfg=json.loads(config_path.read_text());job=cfg.get('job',{})
        for h in json.loads(path.read_text()):
            audit=h.get('gradient_audit',{})
            for factor,d in audit.get('control_variate',{}).items():
                rows.append(dict(task=cfg['task'],method=job.get('name',path.parent.name),seed=cfg.get('update_seed'),step=h['step'],factor=factor,n=audit.get('n'),**d))
    if rows:
        with (EXP/'results/gradient_drift.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    methods={'fast_cv4_cache':'CV4','fast_cv8_cache':'CV8','fast_cv16_cache':'CV16'}
    for i,task in enumerate(['paws','mnli']):
        for j,factor in enumerate(['A','B']):
            ax=axes[i,j]
            for method,label in methods.items():
                data=collections.defaultdict(list)
                for r in rows:
                    if (r['task'],r['factor'],r['method'])==(task,factor,method):data[r['step']].append(r['cv_residual_variance_ratio'])
                x=sorted(data)
                if not x:continue
                mean=np.array([np.mean(data[t]) for t in x]);sd=np.array([np.std(data[t],ddof=1) if len(data[t])>1 else 0 for t in x])
                line=ax.plot(x,mean,marker='o',label=label)[0]
                ax.fill_between(x,mean-sd,mean+sd,color=line.get_color(),alpha=.12)
            ax.axhline(1,color='black',linestyle='--',linewidth=1)
            ax.set_title(f'{task.upper()} — LoRA factor {factor}')
            ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256],[32,64,128,256]);ax.grid(alpha=.2)
            if j==0:ax.set_ylabel('Centered residual / true variance')
            if i==1:ax.set_xlabel('Update step')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Does the fixed predictor reduce gradient-estimation variance?',y=.98)
    fig.text(.5,.01,'16 held-out examples per checkpoint; descriptive only. Mean ± seed SD. Below 1 indicates variance reduction.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.96]);folder=EXP/'results/figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'gradient_drift.{ext}',dpi=170)
    log(dict(diagnostic_rows=len(rows),scope='Small held-out diagnostic sample; cached-body methods may report fresh-feature diagnostics and are excluded from this plot. No test readouts or selection.'))

if __name__=='__main__':main()
