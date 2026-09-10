"""Check explicit failures in reference-guided targets, without trusting the reference."""
import math
import sympy as sp
from common import *
p,q=sp.symbols('p q');det=sp.det(sp.Matrix([[p,12,23],[q,19,20],[1,1,1]]));assert sp.expand(det)==-p+11*q-197
solutions=[]
for sign in [-1,1]:
 point=sp.solve([q+5*p-107,det-sign*140],[p,q]);solutions.append({str(k):str(v) for k,v in point.items()})
assert sorted(int(x['p'])+int(x['q']) for x in solutions)==[27,47]
height2=sp.Rational(150**2+130**2-140**2,2);assert height2==9900 and height2!=14000
good_pairings=math.factorial(9)+math.comb(10,5)*math.factorial(4)**2//2
probability=sp.Rational(good_pairings,math.factorial(10));assert probability==sp.Rational(3,25)
zero=0;growth=0;cycles=0;max_steps=0
for a in range(1,11):
 for b in range(1,11):
  for c in range(1,11):
   state=(a,b,c);seen=set()
   for step in range(100):
    if 0 in state:zero+=1;break
    if state[1]-state[0]>=2 and state[2]-state[1]>=2:growth+=1;break
    if state in seen:cycles+=1;break
    seen.add(state);x,y,z=state;state=(y,z,z*abs(y-x))
   else:raise AssertionError('Uncertified recurrence case')
   max_steps=max(max_steps,step)
assert zero==494 and zero+growth+cycles==1000
write(S/'audits/guided_counterexamples.json',dict(method='Exact symbolic/rational/integer checks of specific target errors. Conditional reviewed sample, not population prevalence.',cases={
 '803':dict(candidate_general_rule_implied_period='pi for cos(x/2)',f_zero=1,f_at_pi=0,conclusion='The introductory scaling rule is false even though the later4pi result is correct.'),
 '1510':dict(correct_determinant=str(det),valid_vertices=solutions,largest_sum=47,candidate_vertices=[[33,-58],[313,-1458]],candidate_actual_areas=[str(abs(det.subs({p:x,q:y}))/2) for x,y in [(33,-58),(313,-1458)]],reference_caveat='Raw reference contains a sign typo; the determinant calculation above is from the problem itself.'),
 '1430':dict(correct_50th_coefficient=-101,candidate_coefficient=49,correct_range_width=str(sp.Rational(8080,199)),candidate_own_width=str(sp.Rational(3920,199))),
 '852':dict(sqrt_1440_equals30=False,triangle_30_30_78_valid=30+30>78,candidate_triangle_area_coefficient=str(sp.Rational(30*30,4)),claimed_triangle_coefficient=405,correct_triangle_coefficient=sp.Rational(48*30,4).__str__(),candidate_405_minus441=405-441,claimed_difference=-81),
 '394':dict(candidate_prime_product=7*2**4*3*5,required_product=math.factorial(7),correct_digit_product=9*8*7*5*2),
 '374':dict(correct_height_squared=str(height2),candidate_height_squared=14000,schematic_coordinate_arithmetic=1.5+1,claimed_distance=140),
 '543':dict(correct_BM_squared=str(sp.Rational(22,3).__sub__(sp.Rational(20,3))**2+sp.Rational(896,9)),candidate_BM_squared=str(sp.Rational(68,9)**2)),
 '522':dict(single10cycle_count=math.factorial(9),two5cycle_count=good_pairings-math.factorial(9),probability=str(probability),candidate_own_terminal_fraction=str(sp.Rational(13825,math.factorial(10)))),
 '246':dict(certified_zero_sequences=zero,certified_nonzero_growth=growth,certified_nonzero_cycles=cycles,max_steps=max_steps,growth_certificate='If consecutive positive terms a,b,c satisfy b-a>=2 and c-b>=2, next d=c*(b-a)>=2c and d-c>=2, so this property persists forever and zero is impossible.',candidate_own_total=90+90+16,claimed_final=494)
 }))
print('Guided counterexamples checked; recurrence all1000 cases certified:',zero,growth,cycles)
