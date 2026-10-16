"""Depth-compressed causal teacher suffix with inherited pretrained Qwen blocks."""
from token_critic import TokenCritic
from loss_critic import torch,nn,sdpa_kernel,SDPBackend
from pathlib import Path

class CopiedSuffix(TokenCritic):
    def __init__(self,dim,label_tokens,qwen_config,source_layers=None,prefix_layer=8,kind='copied_suffix'):
        nn.Module.__init__(self)
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2MLP,Qwen2RMSNorm,Qwen2RotaryEmbedding
        config=Qwen2Config(**qwen_config);config._attn_implementation='sdpa';assert config.hidden_size==dim
        source_layers=source_layers or [18,27];self.cfg=dict(dim=dim,label_tokens=label_tokens,qwen_config=qwen_config,source_layers=source_layers,prefix_layer=prefix_layer,kind=kind);self.input_field='h';self.pretrained_readout=True
        self.register_buffer('label_tokens',torch.tensor(label_tokens,dtype=torch.long));self.register_buffer('target_scale',torch.tensor(1.));self.register_buffer('loss_scale',torch.tensor(1.));self.register_buffer('readout',torch.zeros(len(label_tokens),dim));self.register_buffer('readout_norm',torch.ones(dim))
        self.prefix_mlp=Qwen2MLP(config);self.prefix_norm=Qwen2RMSNorm(dim,config.rms_norm_eps);self.layers=nn.ModuleList([Qwen2DecoderLayer(config,i) for i in source_layers]);self.final_norm=Qwen2RMSNorm(dim,config.rms_norm_eps);self.rotary=Qwen2RotaryEmbedding(config)

    @torch.no_grad()
    def initialize_readout(self,model_path):
        from safetensors import safe_open
        mapping={};state=self.state_dict();prefix=self.cfg['prefix_layer']
        for key in state:
            if key.startswith('prefix_mlp.'):mapping[key]=f'model.layers.{prefix}.mlp.'+key.split('.',1)[1]
            elif key.startswith('prefix_norm.'):mapping[key]=f'model.layers.{prefix}.post_attention_layernorm.'+key.split('.',1)[1]
            elif key.startswith('final_norm.'):mapping[key]='model.norm.'+key.split('.',1)[1]
            elif key.startswith('layers.'):
                _,idx,tail=key.split('.',2);mapping[key]=f'model.layers.{self.cfg["source_layers"][int(idx)]}.'+tail
        found=set();readout=None
        for path in Path(model_path).glob('*.safetensors'):
            with safe_open(str(path),framework='pt',device='cpu') as f:
                keys=set(f.keys())
                for destination,source in mapping.items():
                    if source in keys:state[destination].copy_(f.get_tensor(source));found.add(destination)
                if 'lm_head.weight' in keys:readout=f.get_tensor('lm_head.weight')[self.label_tokens.cpu()]
                elif readout is None and 'model.embed_tokens.weight' in keys:readout=f.get_tensor('model.embed_tokens.weight')[self.label_tokens.cpu()]
        assert found==set(mapping) and readout is not None
        self.readout.copy_(readout);self.readout_norm.copy_(self.final_norm.weight)

    def token_logits(self,d,h):
        L=h.shape[1];positions=torch.arange(L,device=h.device)[None,:];mask=torch.zeros(len(h),1,L,L,device=h.device,dtype=h.dtype);forbidden=torch.ones(L,L,device=h.device,dtype=torch.bool).triu(1)[None,None]|~d['valid'][:,None,None,:];mask=mask.masked_fill(forbidden,float('-inf'))
        h=h+self.prefix_mlp(self.prefix_norm(h));rotary=self.rotary(h,positions)
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:h=layer(h,attention_mask=mask,position_ids=positions,position_embeddings=rotary,use_cache=False)
        return self.final_norm(h)@self.readout.T
