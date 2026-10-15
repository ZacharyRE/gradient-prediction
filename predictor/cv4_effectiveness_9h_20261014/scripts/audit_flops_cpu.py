"""Small arithmetic sanity checks for native/fused FLOP accounting."""
from common import EXP,write
from flop_accounting import *
torch.set_num_threads(4);torch.manual_seed(42)
x=torch.randn(2,3,requires_grad=True);w=torch.randn(3,4,requires_grad=True)
with NativeFlopCounterMode(display=False,custom_mapping=FUSED_MAPPING) as c:
    (x@w).sum().backward()
assert c.get_total_flops()==3*2*2*3*4,c.get_total_flops()
linear=c.get_total_flops()
layer=torch.nn.TransformerEncoderLayer(16,4,32,0.,'gelu',batch_first=True,norm_first=True).eval()
input=torch.randn(3,7,16)
with torch.no_grad():expected=layer(input)
with torch.no_grad(),NativeFlopCounterMode(display=False,custom_mapping=FUSED_MAPPING) as c,OperatorCoverage() as coverage:
    actual=layer(input)
expected_flops=8*3*7*16*16+4*3*7*7*16+4*3*7*16*32
assert c.get_total_flops()==expected_flops,(c.get_total_flops(),expected_flops)
assert torch.equal(actual,expected)
result=dict(passed=True,linear_forward_backward_flops=linear,encoder_matrix_flops=c.get_total_flops(),encoder_output_bitwise_equal=True,coverage=coverage_record(coverage,c))
write(EXP/'results/flop_counter_cpu_audit.json',result)
print({k:v for k,v in result.items() if k!='coverage'})
