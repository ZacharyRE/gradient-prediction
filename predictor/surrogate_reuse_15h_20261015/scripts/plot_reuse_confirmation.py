"""Three-start development curves; incomplete replication cells are omitted."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=[('oracle32_lr0.001','Oracle32, LR0.001','#1e293b','-'),('oracle32_lr0.0003','Oracle32, LR0.0003','#64748b','--'),('subset16_lr0.001','Oracle16, LR0.001','#a16207','-'),('cv16_lr0.001','Original CV16, LR0.001','#0284c7','-'),('state256_cv16_lr0.001','State256 CV16, LR0.001','#059669','-'),('zero_every4_lr0.001','Zero/Adam, true every4, LR0.001','#9333ea','-')]

def main():
    fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharey='row');plotted=0
    development=json.loads((EXP/'results/development_summary.json').read_text());warm={r['warmup_seed']:r['accuracy'] for r in development if r['method']=='warmup' and r['warmup_seed'] in [301,311,321]};assert len(warm)==3
    for row,(population,title) in enumerate([('fullpool','Full adaptation pool: 4,097 examples'),('candidate','Candidate 0: 1,024 examples')]):
        path=EXP/'results'/f'{population}_confirmation_summary.json'
        data=json.loads(path.read_text())['rows'] if path.exists() else []
        for method,label,color,style in METHODS:
            rr=sorted([r for r in data if r['method']==method and r['complete']],key=lambda r:r['steps'])
            if not rr:continue
            plotted+=1;y=[r['mean_accuracy_pct'] for r in rr]
            for col,xkey in enumerate(['steps','mean_preparation_plus_online_seconds']):
                axes[row,col].plot([r[xkey] for r in rr],y,marker='o',ms=4,lw=1.6,color=color,ls=style,label=label)
        for col in range(2):
            ax=axes[row,col];ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.15);ax.set_title(title,fontsize=11)
            ax.axhline(float(np.mean(list(warm.values()))),color='#9ca3af',ls=':',lw=1.2,label='Warmup baseline')
            ax.set_xlabel('Adaptation updates after warmup' if col==0 else 'Preparation + online updates (seconds)')
            if col==0:ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512],labels=['32','64','128','256','512']);ax.set_ylabel('Development accuracy (%)')
    legend={}
    for ax in axes.flat:
        handles,labels=ax.get_legend_handles_labels();legend.update(zip(labels,handles))
    if legend:fig.legend(list(legend.values()),list(legend),loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,.045),frameon=False)
    fig.suptitle('Development accuracy and cost across adaptation horizons',fontsize=14)
    fig.text(.06,.015,'Every plotted point averages the same3 development starts. Exploratory timing excludes model startup, offline predictor preparation and evaluation.\nFull input sequences retained; greedy generation permits256 new output tokens. Development curves are not heldout confirmation.',fontsize=8,color='#475569')
    fig.tight_layout(rect=[0,.12,1,.96]);out=EXP/'results/figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'three_start_development.png',dpi=170);fig.savefig(out/'three_start_development.pdf');plt.close(fig)
    log(dict(plotted_method_population_pairs=plotted))

if __name__=='__main__':main()
