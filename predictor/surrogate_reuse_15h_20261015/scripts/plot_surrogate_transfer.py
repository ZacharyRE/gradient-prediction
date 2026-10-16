"""Matched-state gradient diagnostics, never a downstream performance claim."""
from common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def label(name):
    if '__' not in name:return 'Original predictor'
    short=name.split('__',1)[1].removesuffix('_p123')
    if short.startswith('refit_h_'):return 'H / '+short.removeprefix('refit_h_')
    if short.startswith('refit_'):return 'Y / '+short.removeprefix('refit_')
    return 'Multi-state / '+short.removeprefix('multistate_')


def main():
    allrows=[];fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    for ax,task in zip(axes,['paws','mnli']):
        rows=json.loads((EXP/'results'/f'surrogate_transfer_{task}.json').read_text())['rows']
        names=list(dict.fromkeys(r['predictor'] for r in rows))
        values=np.full((len(names),4),np.nan)
        for r in rows:
            i=names.index(r['predictor']);j=[0,32,128,256].index(r['teacher_step'])
            values[i,j]=r['metrics']['all']['unit_coefficient_residual_ratio']
            for factor,metrics in r['metrics'].items():
                allrows.append(dict(task=task,predictor=r['predictor'],teacher_step=r['teacher_step'],factor=factor,n=r['n'],**metrics))
        assert np.isfinite(values).all(),'Require every matched teacher-state readout'
        im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2),aspect='auto')
        for i in range(len(names)):
            for j in range(4):
                v=values[i,j];ax.text(j,i,f'{v:.2f}' if v<100 else f'{v:.0f}',ha='center',va='center',fontsize=8,color='white' if v<.35 or v>1.65 else '#222222')
        ax.set_xticks(range(4),labels=['0','32','128','256']);ax.set_yticks(range(len(names)),labels=[label(n) for n in names],fontsize=9)
        ax.set_title(f'{task.upper()} · {rows[0]["n"]} identical held-out examples')
        ax.set_xlabel('Oracle32 teacher update step after warmup')
    fig.colorbar(im,ax=axes,label='Centered residual variance / true-gradient variance',shrink=.65,extend='max')
    fig.suptitle('Do frozen gradient surrogates transfer across training states?\nUnit correction coefficient; lower is better, 1 means no variance reduction',fontsize=13)
    out=EXP/'results/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'surrogate_transfer.{ext}',dpi=170)
    plt.close(fig)
    with (EXP/'results/surrogate_transfer.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(allrows)
    log(dict(matched_diagnostic_rows=len(allrows),note='Color saturates above2; annotated numbers retain actual values. Diagnostic gradients do not fit or select a predictor. No downstream-quality inference from this chart alone.'))


if __name__=='__main__':main()
