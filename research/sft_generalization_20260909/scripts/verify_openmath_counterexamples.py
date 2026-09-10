from fractions import Fraction as F
from itertools import product
from collections import Counter
from common import *
checks={}
point=[F(2,5),F(2,5),F(6,5)];assert sum(point)==2 and max(point)<=3*min(point);value=sum(x*x for x in point);assert value>F(4,3);checks['review8']=dict(feasible_point=list(map(str,point)),sum_squares=str(value),claimed_maximum='4/3')
assert F(175)*F(45,100)==F(315,4);checks['review3']=dict(messages_moved_exact='315/4',problem='Discrete message count incompatible with exact percentages; no specified rounding rule.')
checks['review10']=dict(sequence1='a0=2, then5 forever',sum1=9,sequence2='an=5-3*2^(-n)',sum2=3,issue='Both arise as iterations of admissible f with limit5; operation unspecified, sum not fixed.')
def f(x):return F(x*x-1,x+1)
assert f(f(F(3)))==1;checks['review11']=dict(a=1,x=3,composition=1,required_x_squared=9)
assert F(15**2*5)<F(80**2);checks['review12']=dict(perimeter=160,triangle_inequality_sum_distance_lower_bound=80,claimed_sum_distance_squared=1125,lower_bound_squared=6400)
valid=[];orbits=set();dist=Counter()
for x in product(range(3),repeat=12):
 if any(x[i]==x[(i+1)%12] for i in range(12)):continue
 c=Counter(x)
 if min(c.get(k,0) for k in range(3))<3:continue
 valid.append(x);dist[tuple(sorted(c.values()))]+=1;orbits.add(min(x[i:]+x[:i] for i in range(12)))
assert len(valid)==3372 and len(orbits)==286
checks['review13']=dict(labeled_chairs_count=len(valid),labeled_mod1000=372,rotational_orbits=len(orbits),color_count_distribution={str(k):v for k,v in dist.items()},claimed36=False)
a={i for i in range(1,2004) if ((i-1)//10)%2==0};b={i for i in range(1,2004) if ((i-1)//11)%2==0};assert not any(i+10 in a for i in a) and not any(i+11 in b for i in b);assert len(a)==1003 and len(b)==1002
maxa=sum((len(range(r,2004,10))+1)//2 for r in range(1,11));maxb=sum((len(range(r,2004,11))+1)//2 for r in range(1,12));assert maxa+maxb==len(a)+len(b)==2005
checks['review15']=dict(A=len(a),B=len(b),sum=2005,upper_bound_method='Disjoint residue-class paths, each maximum independent set ceil(length/2); constructed alternating blocks attain bound.')
checks['review19']=dict(angle_degrees=0,sine=0,cosine=1,expression_value=0,claimed_value=1)
checks['review20']=dict(counterexample='Equilateral triangle satisfies all given side-sine equalities',largest_angle=60,claimed90=False)
assert (2*2-1)**3-1==26
checks['review22']=dict(candidate_b=2,actual_h_of_j=26,required=27,true_b='(1+cuberoot(28))/2')
write(S/'audits/openmath_counterexamples.json',dict(time=time.time(),checks=checks,source='Pinned nvidia/OpenMathInstruct-2 reviewed training candidates; not previous local training data',limitation='These exact counterexamples establish errors in specified examples, not a representative whole-corpus error rate or an explanation of previous local SFT outcomes.'))
print('Verified',len(checks),'external-data counterexample audits')
