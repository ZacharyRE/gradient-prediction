"""Record all24 conditioned reviews with executable counterexamples, never release data."""
from fractions import Fraction
from math import comb
from collections import Counter,defaultdict
import sympy as sp
from common import *

path=S/'audits/openmath_judge_sound_review.jsonl';rows=read(path);assert len(rows)==24
reasons=[
 '105 guests minus45 prepared bags gives60; ordinary interpretation includes sponsors among guests.',
 '25*3 +10*4/4=85 minutes.',
 'Inclusion-exclusion:60-12-15+3=36; direct page enumeration agrees.',
 '26 fortnightly sessions*45/60=19.5 hours under conventional52-week year; scheduling endpoints are implicit.',
 '5*15+3*8+2*25=149.',
 '5*30*5000=750000.',
 'Multiplying the product-minus-x by(x-1) gives x^(2^2010)-x^2+x-1, not x^(2^2010)-x. At3 actual value differs from3/2 by -2/(3^(2^2010-1)-1), which is nonzero.',
 'Concurrency is underdetermined; determinant expansion also wrong. Distinct concurrent lines occur for(a,b)=(0,-3) at(-3,-2), and for(1,sqrt(7)-1) at(b/2,b/2), with different ab. Claimed(2,3) even makes second/third lines identical.',
 'The target wrongly restricts free integer t(99) to0. Set t=-10603 in its own construction: p99=0,p100=21207,p101=21208, so the true minimum absolute value is0.',
 'Legendre exponent v3(2004!)=998 is correct. Retain under conventional highest-power-as-exponent wording; the literal numerical power is3^998, not998. Do not count this as a certified literal-power answer.',
 'All three conditional cases and sum3/10 are correct; different colored interpreted as balls of various colors, consistent with explicit3red count.',
 'Claimed root x=1 gives g(1)=-3,g(g(1))=-39 but g(1)+1=-2. Earlier y=1 also is not a root:2-6-1=-5. Exact degree9 real-root count is9, not claimed1.',
 'Nonnegative additivity gives f2=2f1,f4=4f1,f5=5f1, hence7.',
 'One of each color:3*4*5/choose(12,3)=3/11; complementary cases total160 consistently.',
 'Enumerate y=0..22 and x=45-2y;23 pairs.',
 'Magnitude equation2r^2+2r-9=0 has discriminant76>0, hence2 real solutions.',
 'Monic cubic with3given distinct pole locations equals x(x-2)(x+3);e=1,f=-6. Extra numerator-degree remark is unnecessary and does not alter valid denominator argument.',
 'g2=0 and g0=0 give0.',
 '(3*(62+43)-37-41)/3=79.',
 'Remaining5*500+7*200=3900 under the problem stipulated return convention, not a real retailer policy claim.',
 '15*20+15*25=675.',
 '400*(3/4)*(4/5)=240.',
 '4000*(3/4)/10=300 hours under ordinary initially-empty tank assumption.',
 '5+10+20=35.',
]
excluded={6,7,8,11};records=[];groups=defaultdict(Counter)
for i,(row,reason) in enumerate(zip(rows,reasons)):
 assert row['judge_label']=='sound'
 decision='exclude_substantive_math_or_problem_error' if i in excluded else 'retain_with_explicit_reason'
 if i in [3,9,10,16,19,22]:decision='retain_with_wording_or_convention_note'
 records.append(dict(review_index=i,problem_sha256=row['problem_sha256'],target_sha256=row['target_sha256'],review_group=row['review_group'],decision=decision,reason=reason))
 groups[row['review_group']]['exclude' if i in excluded else 'retain']+=1

x,T=sp.symbols('x T');actual=(3*T-7)/(2*(T-1));difference=sp.factor(actual-sp.Rational(3,2));assert difference==-2/(T-1)
for n in range(1,9):assert sp.expand((x-1)*sp.prod(x**(2**k)+1 for k in range(n)))==x**(2**n)-1
concurrency=[]
for a,b,xx,yy in [(0,-3,-3,-2),(1,sp.sqrt(7)-1,(sp.sqrt(7)-1)/2,(sp.sqrt(7)-1)/2)]:
 residuals=[sp.simplify(xx+a*yy-b),sp.simplify(2*xx+b*yy-3*a),sp.simplify(a*xx+3*yy-2*b)]
 assert residuals==[0,0,0]
 normals=[sp.Matrix([1,a]),sp.Matrix([2,b]),sp.Matrix([a,3])]
 assert all(sp.simplify(sp.det(sp.Matrix.hstack(normals[i],normals[j])))!=0 for i in range(3) for j in range(i))
 concurrency.append(dict(a=str(a),b=str(b),x=str(xx),y=str(yy),ab=str(a*b),all_three_lines_distinct=True))
polynomial=sp.expand((x-100)*((x-101)*(-10603)+1)+21207)
values={str(i):int(polynomial.subs(x,i)) for i in [99,100,101]};assert values=={'99':0,'100':21207,'101':21208}
g=2*x**3-5*x;equation=sp.expand(g.subs(x,g)-g-1);assert equation.subs(x,1)==-37
root_count=int(sp.Poly(equation,x).count_roots(-sp.oo,sp.oo));assert root_count==9
assert sum(i%4!=0 and i%5!=0 for i in range(1,61))==36
assert sum(2004//3**k for k in range(1,8))==998
assert Fraction(3*4*5,comb(12,3))==Fraction(3,11)
assert sum(xx+2*yy==45 for xx in range(46) for yy in range(23))==23
checks=dict(product_identity='(x-1)*product(k=0..n-1)(x^(2^k)+1)=x^(2^n)-1 by telescoping; symbolic finite n1..8 checked.',case6_difference_from_claim=str(difference),case6_T='3^(2^2010-1)>1; no enormous integer constructed',case7_distinct_concurrent_counterexamples=concurrency,case8_integer_polynomial=str(polynomial),case8_values=values,case11_equation_at_claimed_x1=-37,case11_exact_real_root_count=root_count,case9_correct_exponent=998)
write(S/'audits/openmath_judge_sound_review_decisions.json',dict(time=time.time(),review_sha256=sha(path),script_sha256=sha(__file__),records=records,group_counts={k:dict(v) for k,v in groups.items()},exact_checks=checks,scope='24 fixed stratified samples conditioned on judge-sound, excluding original reviewed cases. Four substantive new errors, all in augmented_math_long. Not a prevalence-weighted error estimate, not independent human gold, and retained wording conventions are explicit. No data released.'))

cal=json.loads((S/'audits/openmath_judge_calibration.json').read_text())
write(S/'audits/openmath_training_route_decision.json',dict(time=time.time(),status='quarantined_no_training_release',calibration_sha256=sha(S/'audits/openmath_judge_calibration.json'),followup_decisions_sha256=sha(S/'audits/openmath_judge_sound_review_decisions.json'),reason='Judge-only acceptance is insufficient: three of12 originally excluded targets were calledsound, and4of6 new long augmentedmath judge-sound cases have explicit substantive counterexamples. Prioritize alreadyreleased authenticMATH expansion controls. This does not establish every external target is wrong or that external-data SFT cannot help.',retained_calibration_disagreement='Originalreview18 has correct225 area with swapped shaded/unshaded wording. Judge calls final answer wrong, but both halves equal225; do not describe its rationale as verified.',full12k_inputs_unchanged=True,training_jobs_added=0))
print(json.dumps(dict(group_counts={k:dict(v) for k,v in groups.items()},exact_checks=checks,status='quarantined_no_training_release'),indent=2))
