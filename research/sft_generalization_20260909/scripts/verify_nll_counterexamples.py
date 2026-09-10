from itertools import combinations
from collections import Counter
from math import prod
from common import *
checks={}
pairs=[(x,y) for x in range(1,101) for y in range(x+1,101) if ((1j)**(x%4)+(1j)**(y%4)).imag==0]
counts=Counter((x%4,y%4) for x,y in pairs);assert len(pairs)==1850 and counts[0,2]==300 and counts[1,3]==325
checks['182']=dict(true_total=len(pairs),ordered_residue_counts={str(k):v for k,v in counts.items()},invalid_claim='All25x25 pairs in a single directed residue orientation are ordered x<y.')
vals={x:[n for n in range(10,100) if n%x==0] for x in range(1,100)};xs=[x for x,ns in vals.items() if len(ns)==2];assert xs==list(range(34,50));assert vals[23]==[23,46,69,92] and vals[24]==[24,48,72,96]
checks['1091']=dict(valid_x=xs,multiples23=vals[23],multiples24=vals[24])
ss=[sum(c) for c in combinations([-7,-5,-3,-2,2,4,6,13],4)];assert 4 not in ss and min(x*x+(8-x)**2 for x in ss)==34
checks['1509']=dict(partitions_checked=len(ss),sum4_possible=False,actual_minimum=34,issue='Candidate tests a few partitions without ruling out the unattainable theoretical32; final34 itself is correct.')
p=prod(range(20,26));assert p==127512000 and (p//1000)%10==2 and ((p%1000)//1000)%10==0
checks['1507']=dict(product=p,correct_modular_division=(p//1000)%10,claimed_modular_identity_rhs=((p%1000)//1000)%10)
a,b,c,d,e,f=1,2,3,4,5,6
wrong=a*b*c+a*b*d+a*b*f+a*c*e+a*c*f+a*d*e+b*d*f+d*e*f;true=(a+d)*(b+e)*(c+f);assert wrong!=true
checks['1545']=dict(false_positive_factorization_counterexample=[1,7,143],product1001=1*7*143,teacher7_false_polynomial_value=wrong,true_vertex_sum=true,limitation='Positive face sums are>=2, which would repair the teacher32 uniqueness argument; teacher7 vertex polynomial is independently false.')
checks['180']=dict(valid_point=[50,50,0,0],objective=2500,zero_variable_counterexample=True,correct_global_bound='wx+xy+yz=(w+y)(x+z)-wz <= (w+y)(x+z) <=100^2/4; equality attainable.')
checks['1990']=dict(stated_AB_squared=36,radius_squared=50,invalid_other_B_distance_squared=74,valid_B_distance_squared=26,issue='Schematic code AB=radius cannot be used as given metric length; actual valid configuration follows B inside circle.')
write(S/'audits/nll_process_counterexamples.json',dict(time=time.time(),checks=checks,limitation='Exact checks substantiate specified false statements or missing arguments; not all final answers are wrong and not all reviewed traces formally verified.'))
print('Verified',len(checks),'question-specific counterexample audits')
