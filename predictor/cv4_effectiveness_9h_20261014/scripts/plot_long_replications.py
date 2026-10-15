"""Show complete replicated long curves, with a common recipe across seeds."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows=json.loads((EXP/'results/development_cohorts.json').read_text())['rows']
    choices={
        'paws':[('long_oracle32_lr0.0003','Oracle32, LR 3e-4'),
                ('long_oracle8_lr0.0003','Oracle8, LR 3e-4'),
                ('long_cv8_lr0.0003','CV8, LR 3e-4')],
        'mnli':[('long_oracle32_lr0.0003','Oracle32, LR 3e-4'),
                ('long_oracle8_lr0.001','Oracle8, LR 1e-3'),
                ('long_oracle8_lr0.0001','Oracle8, LR 1e-4'),
                ('long_cv8_lr0.001','CV8, LR 1e-3')],
    }
    colors=['#253b53','#b37523','#76954d','#2176ae']
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.5),sharey=False)
    for ax,(task,methods) in zip(axes,choices.items()):
        for index,(method,label) in enumerate(methods):
            selected=sorted([r for r in rows if r['task']==task and r['method']==method and r['n_seeds']==3],key=lambda r:r['steps'])
            assert selected,(task,method,'Require completed three-seed replication')
            x=np.array([r['steps'] for r in selected]);y=np.array([r['dev_accuracy_mean'] for r in selected]);sd=np.array([r['dev_accuracy_seed_sd'] for r in selected])
            color='#2176ae' if method.startswith('long_cv') else colors[index]
            ax.plot(x,y,'o-',label=label,color=color,linewidth=1.8,markersize=4)
            ax.fill_between(x,y-sd,y+sd,color=color,alpha=.10,linewidth=0)
        ax.set(title=task.upper(),xlabel='Additional updates after 32-step warmup',ylabel='Development accuracy (%)')
        ax.set_xscale('log',base=2);ticks=[32,64,128,256,512,1024];ax.set_xticks(ticks,labels=[str(t) for t in ticks])
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Long-run quality must replicate across seeds',fontsize=14)
    fig.text(.5,.015,'Three paired update/predictor seeds; bands show ±1 sample SD, not confidence intervals. Development selection only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));out=EXP/'results/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'long_replications.{ext}',dpi=180,bbox_inches='tight')
    log(dict(figure='long_replications',tasks=list(choices)))


if __name__=='__main__':main()
