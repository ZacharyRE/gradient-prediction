"""No-optimizer templates; compare cheap single-warmup and multistate sources."""
from predictor_core import *
from position_template import PositionTemplate
from fit_factor import ratio
from updates_legacy import clock

@torch.no_grad()
def main():
    gpu_guard();guard_space();seed(953);tick=clock();source=EXP/'cache/critic_v1';manifest=json.loads((source/'complete.json').read_text());states=[]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True);tokens=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABC']
    for item in manifest['states']:
        name=item['state']['name'];path=source/name;meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False);states.append(dict(name=name,train=load_samples(path/'predictor_train'),dev=load_samples(path/'predictor_dev'),A=meta['A'].cuda(),B=meta['B'].cuda(),s=meta['scale'],receipt=item))
    reserve_worker_cache();results=[]
    for scope in ['w301_step0','all']:
        training=[st for st in states if scope=='all' or st['name']==scope];assert training
        for bins in [0,1,4,8]:
            started=clock();p=PositionTemplate(1536,tokens,bins=bins).cuda();counts=torch.zeros(3,device='cuda')
            for st in training:
                for ids in sample_chunks(st['train'],max_n=16,budget=4096):
                    d=pack([st['train'][i] for i in ids]);_,classes,g=p.compact(d,True)
                    for c in range(3):mask=classes==c;p.templates[c]+=g[mask].sum(0);counts[c]+=mask.sum()
            p.templates/=counts[:,None,None];metrics={};energy={}
            for st in states:
                pred=[[],[]];truth=[[],[]];active_energy=0.;total_energy=0.
                for ids in sample_chunks(st['dev'],max_n=16,budget=4096):
                    d=pack([st['dev'][i] for i in ids]);pp=p.predict_factors(d,st['A'],st['B'],st['s']);tt=raw(d['x'],d['g'],st['A'],st['B'],st['s'])
                    for j in range(2):pred[j].append(pp[j]);truth[j].append(tt[j])
                    active_energy+=float((d['g'].square()*d['active'][...,None]).sum());total_energy+=float(d['g'].square().sum())
                metrics[st['name']]=ratio(*[torch.cat(v) for v in pred],*[torch.cat(v) for v in truth]);energy[st['name']]=active_energy/max(total_energy,1e-20)
            name=f'mnli_position_{scope}_b{bins}';out=EXP/'models'/name;out.mkdir(exist_ok=True);score=float(np.mean([x['centered_cv_ratio'] for x in metrics.values()]));save_predictor(p,out/'best.pt',dev_score=score)
            write(out/'config.json',dict(scope=scope,bins=bins,source_states=[st['name'] for st in training],source_cache_sha256=sha(source/'complete.json'),source_sha256=source_signature(__file__),train_examples=sum(len(st['train']) for st in training),optimizer_steps=0,meaning='Class/relative-position mean activation-gradient templates multiplied by each example current exact LoRA-input X and current A/B',primary_variant_predeclared='w301_step0,bins4; other variants are ablations'))
            report=dict(complete=True,seconds=clock()-started,mean_centered_cv_ratio=score,states=metrics,answer_position_activation_gradient_energy_fraction=energy,training_label_collection_seconds=sum(r['seconds'] for st in training for r in st['receipt']['audits'] if r['split']=='predictor_train'),teacher_trajectory_cost_required=scope=='all',evaluation_gradient_data_not_used_to_fit_templates=True,best_sha256=sha(out/'best.pt'));write(out/'summary.json',report);results.append(dict(name=name,scope=scope,bins=bins,score=score));log(results[-1])
    write(EXP/'results/position_template_fit.json',dict(complete=True,seconds=clock()-tick,results=results,primary_variant='mnli_position_w301_step0_b4'))

if __name__=='__main__':main()
