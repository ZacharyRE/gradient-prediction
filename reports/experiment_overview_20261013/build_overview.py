"""Build a cross-study overview from published evidence; never selects on test scores.

Run from any directory with Python, NumPy and Matplotlib. No training or evaluation.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = set()
OLD = 'predictor/countdown_predictor_results/scaling-domains-20260930'
EARLY = 'predictor/countdown_predictor_results/scaling-benchmarks-20260926'
EFF = 'predictor/countdown_predictor_results/efficient-test-20261002'


def read(path):
    p = ROOT / path
    SOURCES.add(p)
    if p.suffix == '.csv':
        with p.open(newline='') as f:
            return list(csv.DictReader(f))
    return json.loads(p.read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def num(v, signed=False):
    return '—' if v in (None, '') else format(float(v), '+.2f' if signed else '.2f')


NAMES = {'countdown':'Countdown', 'gsm8k':'GSM8K', 'math':'MATH-500', 'aqua':'AQuA',
         'svamp':'SVAMP', 'arc_easy':'ARC-Easy', 'arc_challenge':'ARC-Challenge',
         'openbookqa':'OpenBookQA', 'boolq':'BoolQ', 'mnli':'MNLI', 'paws':'PAWS',
         'hellaswag':'HellaSwag', 'race':'RACE', 'commonsense_qa':'CommonsenseQA',
         'sciq':'SciQ', 'social_i_qa':'SocialIQA', 'winogrande':'WinoGrande', 'sst2':'SST-2',
         'explore_arc_challenge_pool':'ARC-Challenge (expanded pool)',
         'explore_gsm8k_lowlr':'GSM8K (lower LR)', 'explore_svamp_lowlr':'SVAMP (lower LR)',
         'explore_aqua_answer':'AQuA (answer only)', 'explore_math_lowlr':'MATH-500 (lower LR)'}


def name(t):
    return NAMES.get(t, t.replace('explore_', '').replace('_', ' '))


def method(m):
    return {'cv4_correction4':'CV4', 'cv4_correction4_paws':'CV4',
            'oracle':'Oracle32', 'mean':'Mean',
            'oracle4_correction4':'Oracle4 (tuned)',
            'oracle4_correction4_paws':'Oracle4 (tuned)',
            'oracle4_matched_correction4':'Oracle4 (matched)',
            'oracle4_matched_correction4_paws':'Oracle4 (matched)',
            'frozen_globalfit':'Frozen + Globalfit', 'frozen_batchfit':'Frozen + Batchfit',
            'frozen_label_globalfit':'Frozen + labels + Globalfit',
            'Frozen / global fit':'Frozen + Globalfit',
            'adaptive':'Adaptive', 'frozen':'Frozen', 'periodic8':'Periodic8'}.get(m, m or 'Not run')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


small = read('reports/benchmark_0p5b_20261010/summary.csv')
large = read('reports/benchmark_1p5b_20261011/summary.csv')
extras = read('reports/experiment_overview_20261013/additional_evidence.json')
summary = []
for model, study, records in [('0.5B','2026-10-10',small), ('1.5B','2026-10-11',large)]:
    for r in records:
        w = float(r['warmup_accuracy_pct'])
        summary.append(dict(study=study, model=model, task=r['task'],
            n=r.get('evaluation_n',r.get('n')), base_accuracy_pct=r['base_accuracy_pct'],
            warmup_accuracy_pct=w, oracle_accuracy_pct=w+float(r['oracle_gain_pp']),
            predictor_accuracy_pct=w+float(r['predictor_gain_pp']) if r['predictor_gain_pp'] else '',
            predictor_method=method(r['predictor_method']),
            oracle_seeds=r.get('oracle_seed_count',r.get('update_seeds')),
            predictor_gain_pp=r['predictor_gain_pp'],
            predictor_ci95_low_pp=r['predictor_ci95_low_pp'],
            predictor_ci95_high_pp=r['predictor_ci95_high_pp']))

selected = read(OLD+'/results/selected_configs.csv')
raw_old = read(OLD+'/results/all_final_results.csv')
by_tag = {r['tag']:r for r in raw_old}
old_rows = []
for s in selected:
    model, task = s['condition'].split('_',1)
    r = dict(study='2026-09-30',model='1.5B' if model=='1p5b' else '7B',task=task)
    for m in ['base','warmup','oracle','frozen','adaptive','periodic8']:
        tag = s.get(m+'_tag',s['condition']+'__'+m)
        source = by_tag[tag]
        r[m+'_accuracy_pct'] = 100*float(source['accuracy'])
        r['n'] = source['n']
    chosen = by_tag[s['original_selected_predictor']]
    r.update(predictor_method=method(chosen['method']),predictor_accuracy_pct=100*float(chosen['accuracy']),oracle_seeds=1)
    r['predictor_gain_pp'] = r['predictor_accuracy_pct']-r['warmup_accuracy_pct']
    old_rows.append(r)
summary.extend(old_rows)
early = read(EARLY+'/results/main_results.csv')
for r in early:
    summary.append(dict(study='2026-09-26/'+('gentle' if r['condition'].endswith('gentle') else 'original'),
        model='1.5B' if r['model']=='1p5b' else '7B', task=r['task'], n=r['n'],oracle_seeds=1,
        predictor_method='Adaptive (same-LR comparison)',
        **{m+'_accuracy_pct':100*float(r[m+'_accuracy']) for m in ['base','warmup','oracle','frozen','mean','adaptive']},
        predictor_accuracy_pct=100*float(r['adaptive_accuracy']),
        predictor_gain_pp=100*(float(r['adaptive_accuracy'])-float(r['warmup_accuracy']))))
assert len(summary)==52
write_csv('benchmark_summary.csv',summary)

# Preserve every final variant and seed for the two recent studies, including failed screens.
all_variants=[]
recipes=[]
for model,folder in [('0.5B','benchmark_0p5b_20261010'),('1.5B','benchmark_1p5b_20261011')]:
    all_variants.extend(dict(model=model,study=folder,**r) for r in read('reports/'+folder+'/final_variants.csv'))
    data=read('reports/'+folder+'/selected_recipes.json')
    if model=='0.5B':
        for task,q in data.items():
            for role in ['warmup','oracle','predictor']:
                for r in q.get(role,[]):
                    recipes.append(dict(model=model,study=folder,task=task,role=role,
                        **{k:r.get(k,'') for k in ['method','seed','predictor_seed','lr','steps','tag']}))
    else:
        for q in data['recipes']:
            for role in ['warmup','oracle','predictor']:
                recipes.append(dict(model=model,study=folder,task=q['task'],role=role,
                    method=q.get('predictor_method','') if role=='predictor' else role,
                    lr=q.get(role+'_lr',''),steps=q.get(role+'_steps','')))
write_csv('recent_final_variants.csv',all_variants)
write_csv('recent_selected_settings.csv',recipes)

# Aggregate only identical recipes, retaining the number of evaluated seeds.
groups={}
for r in all_variants:
    if r['method'] in ['base','warmup']: continue
    key=tuple(r[k] for k in ['model','task','method','control_settings','lr','steps'])
    groups.setdefault(key,[]).append(r)
method_rows=[]
for key,rows in groups.items():
    r=dict(zip(['model','task','method','control_settings','lr','steps'],key))
    assert len({v['seed'] for v in rows})==len(rows)
    r.update(seeds=len(rows),mean_accuracy_pct=st.mean(float(v['source_disjoint_accuracy_pct']) for v in rows),
             population_n=rows[0]['source_disjoint_n'])
    warmup=next(v for v in all_variants if v['model']==r['model'] and v['task']==r['task'] and v['method']=='warmup')
    r['gain_over_warmup_pp']=r['mean_accuracy_pct']-float(warmup['source_disjoint_accuracy_pct'])
    method_rows.append(r)
write_csv('recent_method_comparisons.csv',method_rows)
details=['# October predictor variants and controls', '', '[Overview](README.md) · [Machine-readable table](recent_method_comparisons.csv)', '',
         'Every completed final recipe, grouped only when method, LR, stopping and control role match. Single-seed alternatives are not three-seed comparisons. These were selected on development; this table does not promote a method based on final accuracy. Accuracy uses the source-disjoint population (BoolQ 0.5B: 2,713 questions).', '']
for model in ['0.5B','1.5B']:
    details += ['## '+model, '']
    for task in dict.fromkeys(r['task'] for r in method_rows if r['model']==model):
        rows=[r for r in method_rows if r['model']==model and r['task']==task]
        details += ['### '+name(task), '',table(['Method','Control settings','LR','Steps','Seeds','Accuracy (%)','Gain over Warmup (pp)'],[
            [method(r['method']),{'oracle':'Oracle32 recipe','previous_selected_predictor':'Earlier predictor recipe','selected_predictor':'Selected predictor recipe'}.get(r['control_settings'],'—'),f"{float(r['lr']):g}",r['steps'],r['seeds'],num(r['mean_accuracy_pct']),num(r['gain_over_warmup_pp'],True)] for r in rows]),'']
(HERE/'METHODS.md').write_text('\n'.join(details))

profile=read(OLD+'/results/controlled_efficiency.csv')
write_csv('controlled_efficiency.csv',profile)
costs=[]
cost_large=read('reports/benchmark_1p5b_20261011/cost_accounting.json')['selected_recipes']
for model,data in [('0.5B',extras['cost_0p5b']),('1.5B',cost_large)]:
    for task,q in data.items():
        for role in ['selected_oracles','selected_predictors']:
            rows=q.get(role,[])
            if not rows: continue
            assert len({r['method'] for r in rows})==1
            assert len({r['steps'] for r in rows})==1
            r=dict(model=model,task=task,role=role,method=rows[0]['method'],seeds=len(rows),steps=rows[0]['steps'],lr=rows[0]['lr'])
            for key in ['synchronized_update_seconds','online_true_gradient_example_exposures',
                        'label_seconds','calibration_seconds','prediction_update_seconds',
                        'offline_collection_wall_seconds','offline_fit_wall_seconds',
                        'selected_recipe_setup_plus_updates_seconds']:
                vals=[float(v[key]) for v in rows if v.get(key) is not None]
                r[key]=st.mean(vals) if len(vals)==len(rows) else ''
            costs.append(r)
write_csv('recent_costs.csv',costs)

control_rows=[]
for c in extras['cv4_controls']:
    score=c['predictor_minus_control']
    control_rows.append(dict(model=c['model'],task=c['task'],control=c['control'],
        cv4_accuracy_pct=100*c['cv4']['mean_source_disjoint_accuracy'],
        oracle4_accuracy_pct=100*c['oracle4']['mean_source_disjoint_accuracy'],
        cv4_steps=c['cv4']['steps'],oracle4_steps=c['oracle4']['steps'],
        cv4_lr=c['cv4']['lr'],oracle4_lr=c['oracle4']['lr'],
        difference_pp=score['mean_pp'],ci95_low_pp=score['ci95_pp'][0],ci95_high_pp=score['ci95_pp'][1],
        same_total_online_true_gradient_exposures=c['same_total_true_gradient_exposures']))
write_csv('cv4_controls.csv',control_rows)


def recent_table(model, supplemental=False):
    rows=[r for r in summary if r['model']==model and r['study'].startswith('2026-10')
          and r['task'].startswith('explore_')==supplemental]
    return table(['Task','N','Base','Warmup','Oracle32','Predictor','Method','P − W'],[
        [name(r['task'])+(' †' if str(r['oracle_seeds'])=='1' else ''),r['n'],
         num(r['base_accuracy_pct']),num(r['warmup_accuracy_pct']),num(r['oracle_accuracy_pct']),
         num(r['predictor_accuracy_pct']),r['predictor_method'],num(r['predictor_gain_pp'],True)] for r in rows])


def old_table(model):
    rows=[r for r in old_rows if r['model']==model]
    if model=='7B':
        rows += [r for r in summary if r['model']=='7B' and r['task']=='math' and r['study']=='2026-09-26/gentle']
    return table(['Task','N','Base','Warmup','Oracle32','Frozen','Adaptive','Periodic8'],[
        [name(r['task'])+(' *' if r['study']!='2026-09-30' else ''),r['n']]+
        [num(r.get(m+'_accuracy_pct')) for m in ['base','warmup','oracle','frozen','adaptive','periodic8']] for r in rows])


profiles=[]
for condition in dict.fromkeys(r['condition'] for r in profile):
    rows={r['method']:r for r in profile if r['condition']==condition}
    m,t=condition.split('_',1)
    profiles.append([('1.5B' if m=='1p5b' else '7B')+' / '+name(t)]+
                   [f"{float(rows[k]['counted_flops'])/1e12:.3f} / {float(rows[k]['median_seconds']):.3f}" for k in ['oracle','frozen','adaptive']])

cost_table=[]
for model in ['0.5B','1.5B']:
    for p in [r for r in costs if r['model']==model and r['role']=='selected_predictors']:
        o=next(r for r in costs if r['model']==model and r['task']==p['task'] and r['role']=='selected_oracles')
        cost_table.append([model+' / '+name(p['task']),method(p['method']),f"{o['steps']} / {p['steps']}",
            f"{o['synchronized_update_seconds']:.1f}",f"{p['synchronized_update_seconds']:.1f}",
            f"{p['selected_recipe_setup_plus_updates_seconds']:.1f}"])

efficient=read(EFF+'/results/quality_compute.csv')
efficient_table=table(['Method','Steps','Accuracy (%)','Online TFLOPs','Cache','Offline fit + labels','Total TFLOPs'],[
    [r['method'],r['steps'],num(r['accuracy_pct']) if r['full_accuracy_complete']=='True' else 'Not completed']+
    [f"{float(r[k]):.3f}" for k in ['online_tflops','cache_tflops','offline_tflops','total_tflops']] for r in efficient])

md='''# LoRA gradient prediction: results across 0.5B, 1.5B and 7B

**Evidence through October 12, 2026 · consolidated October 13 · Qwen2.5-Instruct**

[Project](../../README.md) · [Exact settings](recent_selected_settings.csv) · [All October methods](METHODS.md) · [Benchmark data](benchmark_summary.csv) · [Costs](recent_costs.csv)

**Gradients are learnable; a reliable training-efficiency advantage is not established.** Frozen prediction is cheap online but can drift and collapse. Adaptive calibration can prevent collapse but is expensive. CV4 produces useful gains on a few tasks, without an established advantage over small-batch true-gradient updates.

![Predictor gains and measured online cost](overview.png)

## Coverage and reading guide

| Model | Evaluated task coverage | Main evidence |
|---|---|---|
| 0.5B | 13 tasks | October benchmark screening; predictors on 10 admitted configurations across 10 tasks; earlier Countdown studies |
| 1.5B | 18 tasks | 9 September tasks + 9 October tasks; separate Oracle, data-scale and long-run follow-ups |
| 7B | 5 tasks | Countdown, GSM8K, MATH, AQuA, ARC-Challenge; no corresponding CV4 benchmark study recorded |

**Base** = unadapted model; **Warmup** = common true-gradient starting checkpoint; **Oracle32** = true gradients on all 32 update examples. Accuracy is **%**; gains are **percentage points**. A positive point estimate does not establish a reliable improvement. Studies have different prompts, scoring, warmups and budgets: compare methods within a study, not absolute scores across studies. MATH-500 and full-MATH are evaluation populations of the same task; ARC difficulties are two task configurations, not independent domains.

## 0.5B: October 10 benchmark study

Development-selected predictor recipes; three update seeds where admitted. **†** denotes a single-seed Oracle screen; predictor not run. BoolQ uses the predeclared 2,713-question source-disjoint subset (full 3,270-question results also retained).

'''+recent_table('0.5B')+'''

<details>
<summary>All five completed supplemental configurations — including unsuccessful ones</summary>

'''+recent_table('0.5B',True)+'''

ARC-Challenge expanded-pool and GSM8K lower-LR recipes were admitted on development/confirmation, then evaluated; they are not additional datasets. The original failures above remain visible. All 18 completed configurations are in the CSV.

</details>

CV4 improves BoolQ/PAWS over Warmup; Frozen improves HellaSwag. Other positive predictor means are uncertain. Improvements over Warmup alone do not show superiority to Oracle4 or to Base: BoolQ's Warmup itself is below Base.

[Original report, intervals and protocol](../benchmark_0p5b_20261010/README.md) · [Every final variant and seed](recent_final_variants.csv)

## 1.5B: October 11–12 benchmark study

Three paired update seeds, one fixed warmup per task. All nine Oracle gains have positive recorded family-adjusted intervals. Only MNLI/PAWS have clearly positive selected-predictor gains; the other seven predictor intervals include zero.

'''+recent_table('1.5B')+'''

MNLI uses **Globalfit + CV4**; PAWS uses the original predictor + CV4. They retain 68% and 57% of Oracle gains, respectively. These are gains relative to Warmup, not claims of matched-quality efficiency.

[Original report and intervals](../benchmark_1p5b_20261011/README.md) · [Data splits and prompt settings](../benchmark_1p5b_20261011/benchmark_data_summary.csv)

## Earlier 1.5B / 7B: independently tuned method comparisons

September 30 study: **32 updates, primary fit/update seed 123/101**. Each column uses its own development-selected LR. This shows Frozen, Adaptive and Periodic8 separately; it does **not** replace the original development-selected predictor strategy with the highest final-test score. The original strategy improves on Warmup in 4/13 conditions; Adaptive alone has positive point estimates in 9/13, but no Adaptive–Warmup or Adaptive–Oracle comparison survives its 13-comparison Holm correction.

### 1.5B: nine earlier tasks

'''+old_table('1.5B')+'''

### 7B: all five tasks

'''+old_table('7B')+'''

*MATH note:* this row is from the September 26 **gentle-warmup** experiment: 32 updates, primary seed, with Oracle/Frozen at the same LR as Adaptive (not independently selected). Its three-seed Adaptive mean is 77.80%; Adaptive–Warmup and Adaptive–Mean intervals both include zero. Full-MATH (4,999 nonoverlapping questions) gives Warmup 75.22%, Adaptive 75.30%, Mean 75.28%; no clear extra benefit.

For 1.5B full-MATH under that earlier protocol, Warmup is 52.15%, Adaptive 49.27%, Mean 49.69%. The population includes MATH-500 and is not an independent replication. Different MATH protocols should not be merged.

[September method comparisons and statistical scope](../../predictor/countdown_predictor_results/scaling-domains-20260930/README.md) · [September 26 original/gentle results and repeated seeds](../../predictor/countdown_predictor_results/scaling-benchmarks-20260926/README.md) · [All 52 overview configurations](benchmark_summary.csv)

## What the methods mean

| Method / modification | What changes | Evidence so far |
|---|---|---|
| Frozen | Fit predictor offline; no online backbone backward | Cheap online; occasional short-run gains; long-run drift |
| Adaptive | Refresh true gradients for 4 current + 16 selection examples; train/select predictor each step | Can prevent collapse; no stable cross-task incremental benefit; high runtime |
| Periodic8 / fixed calibration | Calibrate less often or for a fixed number of epochs | Tested cost/stability compromises; no established general advantage |
| CV4 | Predict batch gradients, correct with true-minus-predicted gradients on 4 examples; predictor frozen online | Useful MNLI/PAWS/BoolQ gains; must beat Oracle4, not just Warmup |
| Globalfit | Replace per-example gradient-energy denominators with fixed training-set A/B scales; change checkpoint metric too | Helps some fits; e.g. 0.5B OpenBookQA Frozen 50.07% → 58.07%, Warmup 57%; LR also retuned |
| Batchfit | Add error on the aggregate batch gradient and use it in checkpoint selection | BoolQ aggregate alignment improved without a demonstrated downstream gain |
| Other tested changes | X vs Y, labels/masks/position, residual features, larger networks, multi-state supervision, multiple layers, history carry/reset, feedback and learned preconditioning | Static improvements do not consistently transfer to downstream accuracy; protocols differ |
| Mean / MeanM controls | Fresh small-sample mean / fixed offline dense gradient matrix | Frequently competitive; separate predictor benefit from common gradient direction |

Early direct-gradient work also tested Ridge, MLP, low-rank bottlenecks, pooling, layers/modules/ranks and target-alignment ranking. These are gradient/data-selection studies, not the same downstream training experiment. [Early methods](../../predictor/single_layer/README.md) · [Countdown history](../../predictor/countdown_predictor_results/HISTORY.md)

## Does CV4 beat Oracle4?

Three paired seeds. Differences are **CV4 − Oracle4** with nominal seed/source-group 95% intervals, conditional on fixed warmup and selected recipes; they are not equivalence/noninferiority tests.

**Independently tuned** lets Oracle4 choose its own LR and stopping step on development data. **Matched to CV4** fixes those choices to CV4's selected recipe. The first compares practical methods after tuning; the second isolates the update rule under the same settings. Oracle4 uses only four true gradients per update in both cases.

'''+table(['Model / task','Oracle4 settings','CV4 accuracy','Oracle4 accuracy','Difference [95% CI], pp','CV4 / O4 steps'],[
    [r['model']+' / '+name(r['task']),'Matched to CV4' if r['control']=='oracle4_matched' else 'Independently tuned',
     num(r['cv4_accuracy_pct']),num(r['oracle4_accuracy_pct']),
     f"{num(r['difference_pp'],True)} [{num(r['ci95_low_pp'],True)}, {num(r['ci95_high_pp'],True)}]",
     f"{r['cv4_steps']} / {r['oracle4_steps']}"] for r in control_rows])+'''

0.5B PAWS's independently tuned Oracle4 uses twice as many online true-gradient examples; the matched 128-step comparison has large seed variation. All other rows match total online true-gradient exposures. Predictor offline supervision is additional. On 1.5B, mean paired CV4/Oracle4 online time is **2.35× on MNLI, 2.33× on PAWS**. No row establishes a positive CV4 advantage. [Exact comparisons](cv4_controls.csv) · [1.5B control timing](../benchmark_1p5b_20261011/four_gradient_control_costs.csv)

## Cost: online FLOPs and elapsed time

September 30 controlled profiling on H200; same initial state and main batch32, 2 warmups + 7 randomized timing repetitions. Each cell is **TFLOPs per step / seconds per step** (median time). Adaptive includes gradient-label collection, calibration and selection. **Offline preparation is excluded.** These are first-step profiles, not full-trajectory averages.

'''+table(['Model / task','Oracle32','Frozen','Adaptive'],profiles)+'''

**Frozen:** 0.143–0.156× counted FLOPs and 0.171–0.242× time of Oracle32. **Adaptive:** 0.704–0.904× FLOPs but **2.18–2.97× time**. Less arithmetic did not imply lower latency. FLOPs count matrix/attention arithmetic (multiply-add = 2); some scalar, normalization, softmax, transfer and optimizer work is excluded. Offline FLOPs and 7B MATH profiling were not measured in this study. [All five methods and raw values](controlled_efficiency.csv)

Including offline collection/fitting, the original selected 32-step predictor strategies cost **3.37–19.68×** the same-LR Oracle across all 13 conditions. These are one-use adaptation costs, excluding common warmup; neither amortization nor matched quality is assumed. [Measured trajectories](../../predictor/countdown_predictor_results/scaling-domains-20260930/results/cost_amortization.csv)

<details>
<summary>October studies: all selected predictor trajectories and their actual time costs</summary>

Mean per run over available paired seeds. **O / P steps** may differ because methods select stopping independently. Oracle time is online; P total adds recorded offline collection + fitting to online updates. FLOPs were not profiled for these October benchmark recipes; the September ratios must not be transferred to them.

'''+table(['Model / task','Predictor','O / P steps','Oracle online, s','P online, s','P setup + online, s'],cost_table)+'''

Common warmup, loading, search, generation evaluation and separate online audits are excluded; the recorded offline collector can include validation/diagnostic work. These observed timings are not proof of continuously exclusive GPU execution or production-optimal implementations. Faster trajectories with lower accuracy do not establish speedup at equal quality. [Per-task costs and components](recent_costs.csv) · [Exact learning rates and seeds](recent_selected_settings.csv)

</details>

### Separate 0.5B Countdown efficiency pilot

Small predictor + four true-gradient corrections, calibrated every four steps, with cached frozen-prefix inputs. **Cumulative structural FLOP estimates**, including attention matrix arithmetic and corrected offline costs; not GPU time measurements. Common warmup/search/evaluation excluded. Final accuracy completed on CPU for one training seed only.

'''+efficient_table+'''

The predictor uses about 22.45× less counted work than 1,024-step Oracle32 **at lower quality** (12.89% vs 23.54%). It uses about **17.4% more total work** than matched 256-step ProbeOnly4, whose full accuracy evaluation was not completed. This is not a matched-quality speedup or a result for the newer CV4 benchmark recipes. [Pilot details and limitations](../../predictor/countdown_predictor_results/efficient-test-20261002/README.md)

## Follow-ups that explain the limits

| Study | Recorded result | Meaning |
|---|---|---|
| 0.5B Countdown, original layer8 Adaptive | Warmup 4.49% → Adaptive 7.32%, Oracle 7.45% (3 seeds) | A local positive result; no demonstrated efficiency advantage |
| 0.5B Countdown, joint layers 3/8/19 | Warmup 8.64%; Frozen32 0%; Adaptive32 8.04%; Oracle32 10.73% | Calibration averts collapse without a net gain over its own start |
| 1.5B BoolQ, 2K/4K/6K offline data | Warmup 78.99%; Oracle 81.70%; Adaptive 76.12/75.10/75.39%; Frozen 0% after 256 steps | More data improves initial cosine but does not preserve update quality; fixed LR 3e-4 |
| 1.5B Oracle scope / targets | BoolQ 78.99→82.53%; OpenBookQA 76.40→81.47%; ARC-Easy 86.99→90.47%; ARC-Challenge 75.00→75.55% | Genuine continuation headroom on some tasks; ARC-Easy includes formatting gains; ARC-Challenge uncertain |
| 1.5B GSM8K target-source study | Base 73.09%; teacher-target selected mean 75.08%; continuation gain after 32 steps only +0.68 pp, uncertain | Oracle-only study; better targets do not yet establish reliable post-warmup headroom |
| 1.5B OpenBookQA, old-prompt 128-step follow-up | Warmup 76.20%; Oracle 79.87%; Frozen 76.67%; Adaptive 76.53%; Periodic8 76.00% | Oracle gain exists; predictor retains little of it |
| Frozen long runs, 1.5B/7B | Four 1,024-step cases all below Warmup and same-LR time-budget Oracle | Setup cost amortizes, but quality deteriorates |

For the original 7B damaged-warmup Countdown setting, Adaptive 15.23% beats Frozen 0%, but Base is 37.40%. This supports recovery, not superiority to the unadapted model. Calibration is neither universally useless nor sufficient for efficient learning.

The OpenBookQA 128-step predictor follow-up reuses the existing offline predictor and selects each method's LR on the full development set; its three predictor gain intervals include zero. It has no separately established controlled cost claim. Sources: [Countdown layers](../../predictor/countdown_predictor_results/README.md#joint-and-independent-layer-updates), [BoolQ data scale](../../countdown_predictor_results/boolq-predictor-datascale-20261010/README.md), [Oracle follow-ups and old-prompt bridge](../oracle_improvements_20261007/README.md), [OpenBookQA predictor counts/settings/intervals](additional_evidence.json), [GSM8K targets](../target_supervision_20261005/README.md), [long-run scores/cost](../../predictor/countdown_predictor_results/scaling-domains-20260930/README.md#long-runs-cost-is-amortized-but-accuracy-deteriorates).

## Setup and interpretation boundaries

| Setting | September scale/domain studies | October benchmark studies |
|---|---|---|
| Backbone / adapter | Qwen2.5-Instruct; frozen backbone; layer 8 o_proj; rank = alpha = 64 | Same; task-specific adapters/predictors |
| Update optimizer | Fresh AdamW after warmup; batch 32, weight decay 0, clip 1; FP32 master/BF16 autocast | Same; constant LR, fresh optimizer |
| Warmup | Sep 26: 32 steps, LR 5e-4 or 5e-5; Sep 30: LR 3e-6…5e-4, 8/32 steps | LR 1e-4/5e-4/1e-3, 8/32 steps; one selected seed 201 |
| Predictor | Width 512, depth 2, 8-head bidirectional Transformer; Y + mask/position/log-RMS | Width 512, depth 2; primarily Y-only; separate objective/input variants |
| Offline fitting | Sep 26: 4096 train + 128 dev; Sep 30: 320–2048 train; 100 epochs | Usually ~2048 train + 128 dev for new tasks; 30 epochs; legacy-task counts vary |
| Online calibration | 4 current fit examples + 16 fixed selection examples; LR 1e-4, 30 epochs, choose epoch 0…30 | Same 4+16 scheme; up to 10 epochs; CV4 uses four corrections and no online fitting |
| Update search | Sep 30: LR 3e-6/1e-5/3e-5/1e-4; fixed 32 steps | Oracle LR 3e-5/1e-4/3e-4/1e-3; steps 8/32/128/256; predictor LR factors 0.03/0.1/0.3/1 of Oracle LR; registered extensions separately |
| Selection / repeats | Development LR selection; primary seed plus limited repeats | Positive Oracle mean on development AND separate confirmation for predictor admission; three paired fit/update seeds for admitted candidates |
| Evaluation | Full prompts; greedy answer generation; usually 2048 new output tokens | Full prompts; usually 256 new tokens for short answers, 2048 for math/Countdown; output caps never truncate inputs |

Training and gradient collection use reference answers; accuracy inference receives prompts only. Labels, offline gradients and current model state are not free information. Oracle examples per step measure true-gradient work, not annotation cost. Recent intervals condition on a fixed warmup; earlier tables often show one primary run. Legacy tests have been inspected previously, so this is exploratory evidence. Multiple model sizes remain within one model family. The four shared October tasks reuse splits/prompts and harmonized grading, but selected recipes differ: this is not a controlled model-size-only comparison.

## Research conclusion and next decision

1. **Learnability is supported; stable efficient learning is not yet established.** Gradient cosine, generated-answer accuracy, FLOPs and time are separate outcomes.
2. **State drift matters.** In BoolQ, the 6K predictor's batch cosine on the same held-out questions drops from 0.917 at Warmup to 0.043 after 32 Oracle steps. More same-state data did not fix it.
3. **Predictor value must exceed cheap true-gradient baselines.** CV4's gains over Warmup do not establish added value over Oracle4. Mean controls are also frequently competitive.
4. **Next:** focus on 1.5B MNLI/PAWS; compare Oracle4/8/16/32, Frozen and CV4 with independently tuned stopping on accuracy-versus-total-time curves. Audit estimation error at identical model states along training. Pursue state coverage or cheaper prediction only when that diagnosis supports it; confirm promising settings with new warmups/held-out data before expanding scale.

## Reproducibility

This is an evidence summary, not a new training run. Tables are generated from existing published CSV/JSON artifacts and a compact export of previously recorded controls/costs. No final-test-based reselection is performed. [Source hashes](provenance.json) · [Build script](build_overview.py) · [Additional source evidence](additional_evidence.json).

Rebuild from the repository root with `python reports/experiment_overview_20261013/build_overview.py` (NumPy + Matplotlib). Checkpoints, activation caches, datasets and raw generations are not uploaded by this update.
'''
(HERE/'README.md').write_text(md)

# A compact overview figure; intervals are the original nominal paired intervals.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(16,6.3),gridspec_kw={'width_ratios':[1.2,1.25,.9]})
for ax,model in zip(axes[:2],['0.5B','1.5B']):
    rows=[r for r in summary if r['model']==model and r['study'].startswith('2026-10') and not r['task'].startswith('explore_') and r['predictor_gain_pp']!='']
    source=small if model=='0.5B' else large
    src={r['task']:r for r in source}
    y=np.arange(len(rows))
    for key,color,offset,label in [('oracle','#64748b',-.15,'Oracle32'),('predictor','#007c91',.15,'Selected predictor')]:
        means=np.array([float(src[r['task']][key+'_gain_pp']) for r in rows])
        lo=np.array([float(src[r['task']][key+'_ci95_low_pp']) for r in rows])
        hi=np.array([float(src[r['task']][key+'_ci95_high_pp']) for r in rows])
        ax.errorbar(means,y+offset,xerr=[means-lo,hi-means],fmt='o',ms=4,color=color,capsize=2,label=label)
    ax.set_yticks(y,[name(r['task'])+' · '+r['predictor_method'].replace('Frozen + Globalfit','F+G') for r in rows],fontsize=9)
    ax.invert_yaxis();ax.axvline(0,color='#b0b8c3',lw=1)
    ax.set_title(model+' October: gain over Warmup',fontweight='bold',fontsize=11)
    ax.set_xlabel('Accuracy gain (pp), nominal 95% interval')
    ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=8)
ax=axes[2]
for i,m in enumerate(['frozen','adaptive']):
    rr=[r for r in profile if r['method']==m]
    for offset,key,color,label in [(-.12,'flop_ratio_to_oracle','#007c91','Counted FLOPs'),(.12,'time_ratio_to_oracle','#b65d26','Measured time')]:
        v=[float(r[key]) for r in rr];lo,hi=min(v),max(v)
        ax.plot([lo,hi],[i+offset]*2,color=color,lw=6,solid_capstyle='round',label=label if i==0 else None)
        ax.text(hi+.08,i+offset,f'{lo:.2f}–{hi:.2f}×',va='center',fontsize=9)
ax.axvline(1,color='#64748b',ls='--',lw=1)
ax.set_yticks([0,1],['Frozen','Adaptive']);ax.invert_yaxis();ax.set_ylim(1.6,-.6)
ax.set_xlim(0,4);ax.set_xlabel('Ratio to Oracle32 (1×)')
ax.set_title('September: online step cost',fontweight='bold',fontsize=11)
ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.15)
fig.suptitle('Learnable gradients; no established matched-quality training speedup',fontsize=16,fontweight='bold',y=.99)
fig.text(.5,.025,'Accuracy: selected October recipes; 0.5B shows eight primary predictor tasks. Cost: 13 September conditions, offline excluded.\nSeparate studies and protocols; lower step cost does not imply equal downstream quality.',ha='center',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.11,1,.94],w_pad=2)
fig.savefig(HERE/'overview.png',dpi=180)
fig.savefig(HERE/'overview.pdf')
plt.close(fig)

provenance={'as_of':'2026-10-13','scope':'Derived publication; no new experiments, no test-based reselection',
    'sources':[{'repository_path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SOURCES)],
    'counts':{'benchmark_summary_rows':len(summary),'recent_final_variants':len(all_variants),'profile_rows':len(profile),'recent_cost_rows':len(costs),'cv4_contrasts':len(control_rows)}}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance['counts']))
