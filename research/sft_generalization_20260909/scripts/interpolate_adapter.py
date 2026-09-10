"""Create an explicitly derived base/SFT interpolation by scaling LoRA alpha."""
import argparse,shutil
from common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--factor',type=float,required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args()
 assert 0<=a.factor<=1
 config=json.loads((a.source/'adapter_config.json').read_text());assert not config.get('use_dora') and not config.get('use_rslora') and not config.get('alpha_pattern') and not config.get('modules_to_save') and config.get('bias','none')=='none'
 alpha=config['lora_alpha']*a.factor;assert alpha==int(alpha);config['lora_alpha']=int(alpha)
 root=S/'results/training'/a.name;dest=root/f'epoch{a.epoch}';dest.mkdir(parents=True,exist_ok=False)
 shutil.copy2(a.source/'adapter_model.safetensors',dest/'adapter_model.safetensors');write(dest/'adapter_config.json',config)
 manifest=dict(kind='derived_weight_interpolation_not_new_training',source=str(a.source.resolve()),source_model_sha256=sha(a.source/'adapter_model.safetensors'),source_config_sha256=sha(a.source/'adapter_config.json'),factor=a.factor,formula='W = W_base + factor * (alpha_train/r) B A',script_sha256=sha(__file__),time=time.time())
 write(dest/'derivation.json',manifest)
 write(root/'manifest.json',dict(kind='derived_weight_interpolation_not_new_training',checkpoints={f.parent.name:json.loads(f.read_text()) for f in root.glob('epoch*/derivation.json')}))
 write(dest/'ready.json',dict(name=a.name+f'_epoch{a.epoch}',adapter=str(dest.resolve()),epoch=a.epoch,derivation=manifest,time=time.time()))
 print(str(dest))
if __name__=='__main__':main()
