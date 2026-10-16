from predictor_core import *
import argparse,copy

METHODS=['oracle','oracle20','mean','frozen','adaptive','periodic8']

def clock():
 torch.cuda.synchronize();return time.perf_counter()

def calibrate(p,train,valid,A,B,s,epochs=30,lr=1e-4):
 A=A.detach();B=B.detach();opt=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=.01)
 before=diagnose(p,valid,A,B,s);bestscore=before['selection_score'];bestepoch=0
 best={k:v.detach().clone() for k,v in p.state_dict().items()};scores=[dict(epoch=0,score=bestscore)];losses=[]
 for epoch in range(1,epochs+1):
  p.train();opt.zero_grad(set_to_none=True);total=0
  for ix in sample_chunks(train,max_n=4):
   d=pack([train[i] for i in ix]);loss=objective(p,d,A,B,s);(loss*len(ix)/len(train)).backward();total+=float(loss.detach())*len(ix)/len(train)
  nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(total)
  score=diagnose(p,valid,A,B,s)['selection_score'];scores.append(dict(epoch=epoch,score=score))
  if score<bestscore:bestscore=score;bestepoch=epoch;best={k:v.detach().clone() for k,v in p.state_dict().items()}
 p.load_state_dict(best);p.eval()
 return dict(selected_epoch=bestepoch,before=before,after=diagnose(p,valid,A,B,s),scores=scores,losses=losses)

@torch.no_grad()
def predict_pair(p,rows,A,B,s):
 # Calibration rows also contain true G; prefix-only rows do not. Inference
 # must consume features alone regardless of which source supplied a row.
 rows=[{k:v for k,v in row.items() if k!='g'} for row in rows]
 ga=torch.zeros_like(A);gb=torch.zeros_like(B)
 for ix in sample_chunks(rows):
  d=pack([rows[i] for i in ix]);pa,pb=raw(d['x'],p(d),A,B,s);ga+=pa.sum(0);gb+=pb.sum(0)
 n=sum(r['n'] for r in rows);return ga/n,gb/n

@torch.no_grad()
def adam_delta(opt,parameters,gradients):
 group=opt.param_groups[0];beta1,beta2=group['betas'];norm=flat(gradients).norm();clip=min(1.,1./(float(norm)+1e-6));parts=[]
 for w,g in zip(parameters,gradients):
  state=opt.state[w];step=int(state.get('step',0))+1;g=g*clip;m=state.get('exp_avg',torch.zeros_like(w))*beta1+g*(1-beta1);v=state.get('exp_avg_sq',torch.zeros_like(w))*beta2+g.square()*(1-beta2)
  parts.append(-group['lr']*(m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+group['eps']))
 return flat(parts)

def one_step(method,m,tok,mod,cap,p,ds,val,ids,cal,opt,step,cal_epochs=30,micro=4,mean_matrix=None):
 A,B,s=weights(mod);row={};t0=clock()
 row['true_gradient_examples']=0;row['true_gradient_tokens']=0;row['true_gradient_input_tokens']=0
 row['main_batch_input_tokens']=sum(ds.lengths[i] for i in ids)
 if method=='train_mean_matrix':
  assert mean_matrix is not None
  opt.zero_grad(set_to_none=True)
  with torch.no_grad():A.grad=s*B.T@mean_matrix;B.grad=s*mean_matrix@A.T
  row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True);row['model_input_tokens']=0
 elif method in ['oracle','oracle20']:
  used=ids if method=='oracle' else ids[:20]
  row.update(oracle_step(m,tok,mod,ds,used,opt,micro=micro));row['true_gradient_examples']=len(used);row['true_gradient_tokens']=sum(ds[i]['n'] for i in used)
  row['true_gradient_input_tokens']=sum(ds.lengths[i] for i in used);row['model_input_tokens']=row['true_gradient_input_tokens']
 elif method=='mean':
  opt.zero_grad(set_to_none=True);v_ids=list(range(16));N=sum(ds[i]['n'] for i in cal)+sum(val[i]['n'] for i in v_ids)
  for dataset,indexes in [(ds,cal),(val,v_ids)]:
   for ix in chunks(dataset,indexes,max_n=micro):
    bb=batch(dataset,ix,tok);n=sum(dataset[i]['n'] for i in ix);loss=forward(m,bb).loss;(loss*n/N).backward()
  row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
  row['true_gradient_examples']=20;row['true_gradient_tokens']=N
  row['true_gradient_input_tokens']=sum(ds.lengths[i] for i in cal)+sum(val.lengths[i] for i in v_ids);row['model_input_tokens']=row['true_gradient_input_tokens']
 else:
  cal_rows=[];val_rows=[];needs_cal=method in ['adaptive','mean'] or (method=='periodic8' and (step-1)%8==0)
  row['model_input_tokens']=row['main_batch_input_tokens']+(sum(val.lengths[i] for i in range(16)) if needs_cal else 0)
  if needs_cal:
   t=clock();cal_rows,ct,_=collect(m,tok,mod,cap,ds,cal,micro=micro,verify_reconstruction=False);val_rows,vt,_=collect(m,tok,mod,cap,val,list(range(16)),micro=micro,verify_reconstruction=False)
   row['label_seconds']=clock()-t;row['true_gradient_examples']=len(cal_rows)+len(val_rows);row['true_gradient_tokens']=sum(r['n'] for r in cal_rows+val_rows)
   row['true_gradient_input_tokens']=sum(ds.lengths[i] for i in cal)+sum(val.lengths[i] for i in range(16))
   if method!='mean':
    t=clock();row['calibration']=calibrate(p,cal_rows,val_rows,A,B,s,epochs=cal_epochs);row['calibration_seconds']=clock()-t
  t=clock()
  if method=='mean':
   n=sum(r['n'] for r in cal_rows+val_rows);ga=(ct[0]+vt[0])/n;gb=(ct[1]+vt[1])/n
  else:
   # Reuse the four already-captured features when calibration occurred.
   remaining=[i for i in ids if i not in set(cal)] if needs_cal else ids
   rows=cal_rows+features(m,tok,mod,cap,ds,remaining,micro=micro);ga,gb=predict_pair(p,rows,A,B,s)
  opt.zero_grad(set_to_none=True);A.grad=ga.detach();B.grad=gb.detach();row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
  row['prediction_update_seconds']=clock()-t
 row['online_seconds']=clock()-t0
 return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--task',required=True);ap.add_argument('--methods',nargs='+',default=METHODS)
 ap.add_argument('--lrs',nargs='+',type=float,default=[3e-6,1e-5,3e-5,1e-4]);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor',default='predictor');ap.add_argument('--cal-epochs',type=int,default=30);ap.add_argument('--micro',type=int,default=4);ap.add_argument('--condition-suffix',default='');ap.add_argument('--start');ap.add_argument('--mean-matrix');a=ap.parse_args()
 name=a.model+'_'+a.task+a.condition_suffix;root=EXP/'models'/name;root.mkdir(parents=True,exist_ok=True);selection=json.loads((EXP/'configs'/f'start_{a.model}_{a.task}.json').read_text());adapter=a.start or selection['adapter']
 m,tok,mod=setup(a.model,adapter,rng=a.seed);A,B,s=weights(mod);original=[v.detach().clone() for v in [A,B]];basever=versions(m);cap=None
 mean_matrix=None
 if 'train_mean_matrix' in a.methods:
  assert a.mean_matrix and a.condition_suffix,'Supplementary control needs its own output condition and declared matrix'
  matrix_record=torch.load(a.mean_matrix,map_location='cpu',weights_only=False);assert matrix_record['metadata']['start_sha256']==sha(Path(adapter)/'adapter_model.safetensors');assert matrix_record['scale']==s
  assert matrix_record['metadata']['source_sha256']==sha(data(a.task,'predictor_train'))
  mean_matrix=matrix_record['M'].cuda();assert tuple(mean_matrix.shape)==(B.shape[0],A.shape[1]);assert torch.isfinite(mean_matrix).all()
 ds=Dataset(a.task,'update',tok);val=Dataset(a.task,'predictor_dev',tok);dev=Dataset(a.task,'accuracy_dev',tok);auditds=Dataset(a.task,'gradient_test',tok);rng=np.random.default_rng(a.seed);order=[]
 while len(order)<a.steps*32:order.extend(int(i) for i in rng.permutation(len(ds)))
 calrng=np.random.default_rng(a.seed+10000);batches=[order[j*32:(j+1)*32] for j in range(a.steps)];calibrations=[[ids[int(j)] for j in calrng.choice(32,4,replace=False)] for ids in batches]
 entries=[]
 for lr in a.lrs:
  for method in a.methods:
   if deadline(reserve=4*3600):raise RuntimeError('Final evaluation reserve reached')
   out=root/f'{method}_lr{lr:g}_s{a.seed}'
   if (out/'summary.json').exists():entries.extend(json.loads((out/'summary.json').read_text())['entries']);continue
   out.mkdir(exist_ok=False);seed(a.seed)
   if cap is not None:cap.close()
   cap=None if method in ['oracle','oracle20','mean','train_mean_matrix'] else Capture(mod)
   with torch.no_grad():A.copy_(original[0]);B.copy_(original[1])
   p=None if method in ['oracle','oracle20','mean','train_mean_matrix'] else load_predictor(root/a.predictor/'best.pt')[0]
   opt=torch.optim.AdamW([A,B],lr=lr,weight_decay=0);history=[];ee=[];baseline=ce(m,tok,dev,micro=a.micro)
   write(out/'config.json',dict(args=vars(a),method=method,lr=lr,start_adapter=adapter,start_sha256=sha(Path(adapter)/'adapter_model.safetensors'),baseline_ce=baseline,predictor=str(root/a.predictor/'best.pt') if p else None,input_truncation=False))
   for step,(ids,cal) in enumerate(zip(batches,calibrations),1):
    row=one_step(method,m,tok,mod,cap,p,ds,val,ids,cal,opt,step,a.cal_epochs,a.micro,mean_matrix);row.update(step=step,indices=ids,calibration_indices=cal,input_tokens=sum(ds.lengths[i] for i in ids),supervised_tokens=sum(ds[i]['n'] for i in ids))
    if step in [1,8,16,32]:
     dest=out/f'step-{step}';m.save_pretrained(dest);row['dev_ce']=ce(m,tok,dev,micro=a.micro)
     if p is not None and step in [1,8,32]:
      # Post-update held-out audit: never used for calibration or selection.
      # Its real backpropagation/time is explicitly excluded from online cost.
      at=clock();ar,truth,_=collect(m,tok,mod,cap,auditds,list(range(32)),micro=a.micro,verify_reconstruction=False)
      diagnostic=diagnose(p,ar,A,B,s,factor_detail=True);ga,gb=predict_pair(p,ar,A,B,s);n=sum(r['n'] for r in ar)
      diagnostic['same_history_next_adam_delta']=measures(adam_delta(opt,[A,B],[ga,gb]),adam_delta(opt,[A,B],[truth[0]/n,truth[1]/n]))
      diagnostic['state']='after_current_update';row['heldout_gradient_audit']=diagnostic;row['diagnostic_seconds']=clock()-at;row['diagnostic_true_gradient_examples']=32
      m.zero_grad(set_to_none=True)
     e=dict(model=a.model,task=a.task,condition=name,method=method,lr=lr,seed=a.seed,step=step,adapter=str(dest),tag=f'{name}__{method}_lr{lr:g}_s{a.seed}__step{step}',dev_ce=row['dev_ce'],online_seconds=sum(r['online_seconds'] for r in history)+row['online_seconds']);entries.append(e);ee.append(e)
    history.append(row);write(out/'history.json',history);log(dict(condition=name,method=method,lr=lr,step=step,online_seconds=row['online_seconds'],dev_ce=row.get('dev_ce'),selected_epoch=row.get('calibration',{}).get('selected_epoch')))
   if p is not None:save_predictor(p,out/'predictor-final.pt')
   write(out/'summary.json',dict(complete=True,entries=ee,online_seconds=sum(r['online_seconds'] for r in history),true_gradient_examples=sum(r['true_gradient_examples'] for r in history),base_unchanged=versions(m)==basever))
   write(EXP/'configs'/f'update_entries_{name}_s{a.seed}.json',entries)
 if cap is not None:cap.close()
 assert versions(m)==basever

if __name__=='__main__':main()
