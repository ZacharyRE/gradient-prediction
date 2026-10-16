"""Plot recorded development variance ratios and the best scalar diagnostic."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result=json.loads((EXP/'results/scalar_cv_diagnostic.json').read_text())
    rows={r['state']:r for r in result['rows'] if r['predictor']=='mnli_critic_state256'}
    steps=[0,32,128,256,512]
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.8),sharey=True)
    for ax,prefix,title in zip(axes,['oracle32_lr0.001','state256_cv16_lr0.001'],['Along Oracle32 updates','Along State256–CV16 updates']):
        data=[rows['warmup301']]+[rows[f'{prefix}_step{s}'] for s in steps[1:]]
        ax.plot(steps,[r['actual_residual_variance_ratio'] for r in data],'-o',color='#a23b48',label='Recorded predictor')
        ax.plot(steps,[r['best_residual_ratio_with_one_scalar'] for r in data],'--s',color='#236b8e',label='Best single scalar (diagnostic)')
        ax.axhline(1,color='#555555',linestyle=':',label='No predictor')
        ax.set_title(title);ax.set_xlabel('Additional updates after warmup')
        ax.set_xticks(steps);ax.set_ylim(.86,1.15);ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Residual variance / true-gradient variance')
    axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    fig.suptitle('Late-state gradient diagnostics: State256 predictor',fontsize=13)
    fig.subplots_adjust(top=.83,bottom=.25,wspace=.12)
    fig.text(.08,.04,'129 development examples; start 301; LR 0.001. Lower ratios are better; 1 is the no-predictor control.\nThe best scalar uses all true diagnostic gradients and is not a deployable calibration policy.\nThis fixed-state variance calculation is not a downstream-accuracy or trajectory guarantee.',fontsize=9,color='#4c5662')
    out=EXP/'results/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'scalar_cv_diagnostic.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
