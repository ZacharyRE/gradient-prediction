"""Exact forced ABC readout without paying for a learned surrogate or initial anchor."""
from reuse_train import *
from evaluate_behavior import fixed_template_dataset
from surrogate_freeze import ensure_surrogate_freeze
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel,SDPBackend
from types import SimpleNamespace

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--entries',required=True);ap.add_argument('--name',required=True);ap.add_argument('--split',choices=['selection_dev','selection_holdout'],default='selection_dev');a=ap.parse_args();entries=json.loads(Path(a.entries).read_text())
    freeze=ensure_surrogate_freeze(SimpleNamespace(model='1p5b',max_new_tokens=256,adapter_group_size=1,split=a.split),entries)
    out=EXP/'results/behavior'/a.name;assert not out.exists();out.mkdir(parents=True);started=time.time()
    m,tok,mod=setup('1p5b',entries[0]['adapter'],rng=881);A,B,s=weights(mod);cap=Capture(mod);ds=fixed_template_dataset('mnli',a.split,tok);configure(True,8192);cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32);cached=cached_rows(cache,ds)
    tokens=[tok.encode(c,add_special_tokens=False) for c in 'ABC'];assert all(len(t)==1 for t in tokens);class_tokens=torch.tensor([t[0] for t in tokens],device='cuda');head=m.get_base_model().lm_head.weight[class_tokens].detach();gold=torch.tensor(['ABC'.index(r['gold']) for r in ds.rows])
    protocol=dict(args=vars(a),freeze=freeze,entries_sha256=sha(a.entries),full_original_prompts=True,fixed_answer_template='\\boxed{A}',gold_class_never_enters_model_input=True,target='Exact forced-format ABC, no learned readout',cache_seconds=cache.seconds,source_sha256=source_signature(__file__));write(out/'protocol.json',protocol);summary={};reserve_worker_cache()
    for e in entries:
        tick=clock();weights_in=load_file(str(Path(e['adapter'])/'adapter_model.safetensors'))
        with torch.no_grad():A.copy_(next(v for k,v in weights_in.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in weights_in.items() if k.endswith('lora_B.weight')))
        parts=[]
        for ids in chunks(ds,list(range(len(ds))),max_n=32,token_budget=8192):
            d=pack(current_features(mod,cached,ids));bb=pc.batch(ds,ids,tok)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
                hidden=m.get_base_model().model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
                mask=(d['label_ids'][...,None]==class_tokens[None,None,:]).any(-1);assert torch.all(mask.sum(1)==1);where=mask.long().argmax(1);logits=hidden[torch.arange(len(ids),device='cuda'),where]@head.T
            parts.append(logits.float().cpu())
        logits=torch.cat(parts);labels=logits.argmax(1);seconds=clock()-tick
        jsonl(out/f'{e["tag"]}__teacher.jsonl',[dict(sample_id=row['sample_id'],gold=row['gold'],prediction='ABC'[int(label)],logits=logit.tolist(),correct=bool(label==g)) for row,g,label,logit in zip(ds.rows,gold,labels,logits)])
        summary[e['tag']]={'teacher':dict(n=len(ds),accuracy=float((labels==gold).float().mean()),seconds=seconds)};write(out/'summary.json',summary)
    cache.close();cap.close();write(out/'complete.json',dict(complete=True,entries=len(entries),wall_seconds=time.time()-started));log(dict(complete=True,name=a.name))

if __name__=='__main__':main()
