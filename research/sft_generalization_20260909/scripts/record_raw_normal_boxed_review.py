"""Exact arithmetic checks for four prespecified normal-stop boxed failures."""
from fractions import Fraction as F
import sympy as sp
from common import *
review=S/'audits/raw_normal_boxed_review.jsonl';rows=read(review)
notes={
('student',171):'Raw correctly writes4+2(n-1)=130 but boxes65; n64, while65 gives132. Baseline and generated-target student expand the algebra correctly.',
('student',180):'Raw writes correct dot-product expressions but evaluates entries(1,1)as-6 instead of-5 and(3,1)as10 instead of14. Baseline and generated-target student give correct product.',
('teacher7',363):'Raw drops second-column multipliers and returns a=7/2,b=1/8, which do not give an inverse. Correct a=-3,b=1/14. Generated-target teacher final is correct but its proof ALSO contains4*(3/14)=6/14 and an unresolved4/7=1 contradiction; do not label that full proof sound.',
('teacher7',98):'Raw changes variable definitions and jumps to1. Setting u=x/a,v=y/b,w=z/c gives sum3 and pairwise-product sum0, so squared sum9. Baseline and generated-target teacher use valid identity.'}
assert set(notes)=={(r['family'],r['index']) for r in rows}
A=sp.Matrix([[1,1,-2],[0,4,-3],[-1,4,3]]);B=sp.Matrix([[2,-2,0],[1,0,-3],[4,0,0]])
assert A*B==sp.Matrix([[-5,-2,-3],[-8,0,-12],[14,2,-12]])
def inverse_product(a,b):return sp.Matrix([[a,2],[1,4]])*sp.Matrix([[-sp.Rational(2,7),sp.Rational(1,7)],[b,sp.Rational(3,14)]])
good=inverse_product(-3,sp.Rational(1,14));bad=inverse_product(sp.Rational(7,2),sp.Rational(1,8))
assert good==sp.eye(2) and bad!=sp.eye(2)
witness=[F(2),F(2),F(-1)]
assert sum(witness)==3 and sum(1/v for v in witness)==0 and sum(v*v for v in witness)==9
decisions=[dict(family=r['family'],index=r['index'],raw_model=r['raw_model'],source_model=r['source_model'],raw_text_sha256=hashlib.sha256(r['raw_prediction'].encode()).hexdigest(),source_text_sha256=hashlib.sha256(r['generated_target_model_prediction'].encode()).hexdigest(),finding=notes[(r['family'],r['index'])]) for r in rows]
write(S/'audits/raw_normal_boxed_review_decisions.json',dict(time=time.time(),script_sha256=sha(__file__),review_sha256=sha(review),reviewer='AI research assistant, not independent human expert',decisions=decisions,exact_checks=dict(case171=dict(correct_n=64,raw_n=65,raw_substitution=4+2*(65-1)),case180=dict(correct_product=[[int(v) for v in row] for row in (A*B).tolist()]),case363=dict(correct_product=str(good),wrong_product=str(bad),source_model_false_product='4*(3/14)=6/7, not6/14'),case98=dict(witness=['2','2','-1'],sum=3,reciprocal_sum=0,square_sum=9)),scope='Four fixed-random cases within baseline/source-correct, raw-wrong, normal-stop complete-box subset. Not a prevalence estimate or a uniquely identified reasoning-compression mechanism.'))
print('Recorded four cases and exact checks.')
