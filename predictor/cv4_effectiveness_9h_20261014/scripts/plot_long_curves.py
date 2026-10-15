"""Plot all prespecified long-horizon learning-rate curves; no test selection."""
from common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=[('oracle32','Oracle32','#202733','-'),('oracle16','Oracle16','#806858','--'),('oracle8','Oracle8','#8d9197','--'),('oracle4','Oracle4','#b9bcc0',':'),('cv4','CV4','#d97827','-'),('cv8','CV8','#2477ad','-'),('cv16','CV16','#238563','-')]


def main():
    all_rows=list(csv.DictReader((EXP/'results/development_live.csv').open()))
    rows=[r for r in all_rows if r['method'].startswith('long_') and r['dev_accuracy_pct']]
    assert rows,'No completed long-horizon development readouts yet'
    out=EXP/'results/figures';out.mkdir(exist_ok=True)
    with (EXP/'results/long_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for axis_type in ['steps','time']:
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharey='row')
        for row,task in enumerate(['paws','mnli']):
            for col,lr in enumerate([.0001,.0003,.001]):
                ax=axes[row,col]
                for method,label,color,style in METHODS:
                    name=f'long_{method}_lr{lr}'
                    rr=sorted([r for r in rows if r['task']==task and r['method']==name and int(r['seed'])==101],key=lambda r:int(r['steps']))
                    if not rr:continue
                    x=[int(r['steps']) if axis_type=='steps' else float(r['online_seconds'])+float(r['prefix_preparation_seconds']) for r in rr]
                    y=[float(r['dev_accuracy_pct']) for r in rr]
                    ax.plot(x,y,label=label,color=color,ls=style,lw=1.8,marker='o',ms=3)
                ax.set_title(f'{task.upper()} · LR {lr:g}',fontsize=11)
                ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
                if axis_type=='steps':
                    ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512,1024],labels=['32','64','128','256','512','1024']);ax.set_xlabel('Update steps')
                else:ax.set_xlabel('Online + prefix preparation (seconds)')
                if col==0:ax.set_ylabel('Development accuracy (%)')
                if col==1:ax.legend(fontsize=8,loc='lower right',ncol=2)
        fig.suptitle('Qwen2.5-1.5B · effectiveness after step 32',fontsize=15,y=.99)
        fig.text(.05,.025,'Update seed 101, predictor seed 123; fixed warmup. Every plotted checkpoint is shown; no best-point envelope.\nFull inputs retained. Times exclude inherited predictor fitting and require clean replay for final speed claims. Missing lines were not configured.',fontsize=9,color='#525a65')
        fig.tight_layout(rect=[0,.08,1,.96])
        for ext in ['png','pdf']:fig.savefig(out/f'long_curves_{axis_type}.{ext}',dpi=160)
        plt.close(fig)
    log(dict(long_checkpoint_readouts=len(rows)))


if __name__=='__main__':main()
