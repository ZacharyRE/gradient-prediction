"""Live development-only accuracy/time curves; matched available seeds per line."""
from common import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=[('fast_oracle32','Oracle32 · LR 3e−4','#202733','-'),('fast_oracle32_lr0.001','Oracle32 · LR 1e−3','#202733','--'),('fast_oracle8','Oracle8 · LR 3e−4','#92979f',':'),('fast_oracle16','Oracle16 · LR 3e−4','#806858','--'),('fast_cv4_cache','CV4 cached · LR 3e−4','#d97827','-'),('fast_cv8_cache','CV8 cached · LR 3e−4','#2477ad','-'),('fast_cv16_cache','CV16 cached · LR 3e−4','#238563','-')]

def main():
    rows=list(csv.DictReader((EXP/'results/development_live.csv').open()));fig,axes=plt.subplots(2,3,figsize=(15,8.5),sharey='row')
    for row,task in enumerate(['paws','mnli']):
        for method,label,color,style in METHODS:
            rr=[r for r in rows if r['task']==task and r['method']==method and r['dev_accuracy_pct'] and int(r['steps']) in [32,64,128,256]]
            groups={step:{int(r['seed']):r for r in rr if int(r['steps'])==step} for step in [32,64,128,256]}
            common=set.intersection(*(set(g) for g in groups.values()))
            if not common:continue
            data=[]
            for step,g in groups.items():
                aa=np.array([float(g[s]['dev_accuracy_pct']) for s in sorted(common)])
                tt=np.array([float(g[s]['online_seconds']) for s in sorted(common)])
                cc=np.array([float(g[s]['prefix_preparation_seconds']) for s in sorted(common)])
                data.append((step,aa.mean(),aa.std(ddof=1) if len(aa)>1 else 0.,tt.mean(),(tt+cc).mean()))
            data=np.array(data)
            for col,ix in enumerate([0,3,4]):
                ax=axes[row,col];ax.plot(data[:,ix],data[:,1],marker='o',ms=4,lw=1.8,ls=style,color=color,label=f'{label} (n={len(common)})')
                if len(common)>1:ax.fill_between(data[:,ix],data[:,1]-data[:,2],data[:,1]+data[:,2],color=color,alpha=.09,linewidth=0)
        for col in range(3):
            ax=axes[row,col];ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_title(f'{task.upper()} · '+['updates','online update time','online + prefix preparation'][col],fontsize=11,pad=10)
            ax.set_xlabel(['Update steps','Seconds','Seconds (pretrained predictor available)'][col],fontsize=9)
            if col==0:ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256],labels=['32','64','128','256']);ax.set_ylabel('Development accuracy (%)')
        axes[row,2].legend(fontsize=7,loc='lower right',frameon=True,framealpha=.9)
    fig.suptitle('Qwen2.5-1.5B · development curves, not final-test evidence',fontsize=15,y=.99)
    fig.text(.05,.025,'One fixed warmup per task. Lines use common available update seeds; shading is seed SD. All input tokens retained.\nPrefix preparation is charged once; offline fitting and research evaluation are excluded. Screening timings are exploratory; final speed claims require clean replays.',fontsize=9,color='#525a65')
    fig.tight_layout(rect=[0,.08,1,.96]);out=EXP/'results/figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'development_curves.png',dpi=160);fig.savefig(out/'development_curves.pdf');plt.close(fig)
    print(out/'development_curves.png')

if __name__=='__main__':main()
