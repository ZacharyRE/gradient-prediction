"""Record full 32-case review before releasing the prospectively planned scale controls."""
import subprocess, math
from fractions import Fraction
from collections import Counter
import sympy as sp
from common import *

notes={
2707:('retain','Circle area16pi, rectangle32pi, short side8, long side4pi.'),
1963:('retain','Radii1 and4; gray area15pi.'),
2771:('exclude_error','Invents a single transversal and angles around C; equation C=360-90=270 is then changed to90 without justification.'),
6:('exclude_error','Perpendicular bisectors are y=3/2 and x=2, not x=0 and slope4/3. Derives17/50 then jumps to1/2.'),
974:('retain','3:4:5 scaling gives horizontal27*4/5=21.6.'),
1023:('retain','Exterior turning-angle sum360 for the regular convex hexagon gives60. Broad any-polygon phrase read in standard school simple-polygon convention.'),
823:('retain','12edges among15 unordered vertex pairs gives4/5.'),
2136:('exclude_minor','Correct area-difference proof13.5/18=.75, but calls region a trapezoid and includes origin incorrectly. Conservative spatial-description exclusion, not invalid numerical core.'),
2213:('retain','Common base2 yields13x=13.'),
832:('retain','CRT gives18,53 among positive counts below60, sum71. Positivity implicit in cupcake count.'),
1659:('retain','Binomial coefficient28*9/25*1/64=63/400.'),
1376:('retain','s(s-4)=0 and positive side exclude zero.'),
1397:('retain','Reciprocal cotangent relation gives7/4.'),
0:('retain','Euclidean division floor(-200/19)=-11, remainder9; flooring explicitly stated.'),
3140:('retain','492*499=245508, remainder8.'),
789:('retain','Real ratio cube8 gives2, first term5040/16=315.'),
56:('retain','Radius5+8+6=19, diameter38.'),
96:('exclude_error','Invents a central triangle although lines concur; false area equation derives10609/144 before unsupported final144.'),
2382:('exclude_error','Invented line equations/vertices, W=X; derives area1/4 then boxes1/2. Actual diagram square vertices give1/2.'),
2260:('retain','Two routes on each segment throughB plus direct route gives5.'),
2142:('exclude_error','Calls quadrilaterals rectangles, uses false half-base-height formula and BF=2-sqrt2; derives6+4sqrt2 then picks8-4sqrt2 from options.'),
2262:('exclude_minor','Correct complete height/diagonal method, but final repeated QD label should be QC; actual QD=sqrt145, not8. Conservative notation exclusion, not failure of computational method.'),
3184:('retain','Correct sector-minus-triangle calculation12pi-18sqrt3. Central60-degree justification abbreviated; diagram explicitly fixes arcs0..60 and240..300, so no false geometric assertion required.'),
353:('retain','Right-triangle inradius formula yields hypotenuse12(1+sqrt3).'),
2857:('retain','Polynomial identity coefficients all14, sum42.'),
2612:('retain','Expansion cancels linear term,5w²=141.'),
548:('exclude_error','Explicitly says the true equality99+40+5=24+120=144 is not true, then retries same digits. Final145 correct; prior false rejection unrepaired.'),
636:('retain','Present value500000/(1.05)^10 rounds306956.63; mathematical benchmark, not financial recommendation.'),
1946:('retain','Positive conventional nested-radical limit yields golden-ratio y, x=(3+sqrt5)/2; convergence assumed as in reference, not independently established in target.'),
1575:('retain','Complement of three sons is1-20/64=11/16 under stated model.'),
1092:('retain','x(x-5)^2 positive exactly(0,5) union(5,infinity).'),
847:('retain','Scale5; small-leg sum sqrt(25+24)=7; large sum35.')}
review=S/'audits/teacher7_expansion_review.jsonl';rows=read(review)
assert len(rows)==32 and set(notes)=={r['expansion_index'] for r in rows}
dest=S/'audits/teacher7_expansion_training_release.json';assert not dest.exists()
decisions=[dict(expansion_index=r['expansion_index'],target_sha256=hashlib.sha256(r['solution'].encode()).hexdigest(),review_group=r['review_group'],decision=notes[r['expansion_index']][0],reason=notes[r['expansion_index']][1]) for r in rows]
excluded=sorted(i for i,(d,_) in notes.items() if d.startswith('exclude'))
assert len(excluded)==8
write(S/'audits/teacher7_expansion_review_decisions.json',dict(time=time.time(),review_sha256=sha(review),reviewer='AI research assistant; not independent human expert',decisions=decisions,excluded_expansion_indices=excluded,categories=dict(Counter(r['decision'] for r in decisions)),limitation='Equal8-per-stratum selected review, not overall prevalence. Six substantive erroneous assertions and two conservative minor exclusions distinguished; retained targets not certified by independent proof checking.'))
assert 99+40+5==math.factorial(4)+math.factorial(5)==144
assert [n for n in range(100,1000) if n==sum(math.factorial(int(c)) for c in str(n))]==[145]
pts=[(Fraction(3,2),Fraction(3,2)),(1,1),(Fraction(3,2),Fraction(1,2)),(2,1)]
area=abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(pts,pts[1:]+pts[:1])))/2
assert area==Fraction(1,2)
assert Fraction(10609,144)!=144 and Fraction(17,50)!=Fraction(1,2)
write(S/'audits/teacher7_expansion_review_exact_checks.json',dict(time=time.time(),case2771=dict(false_step=[360-90,270-180]),case6=dict(correct_bisectors=['y=3/2','x=2'],rectangle_area=3,triangle_area=6,correct_probability='1/2',target_intermediate='17/50'),case96=dict(target_derived='10609/144',correct_area=(2+3+7)**2),case2382=dict(actual_vertices=[[str(v) for v in p] for p in pts],actual_area=str(area),target_intermediate='1/4'),case2142=dict(target_FI_squared='6+4sqrt2',correct_FI_squared='8-4sqrt2',difference=str(sp.simplify(6+4*sp.sqrt(2)-(8-4*sp.sqrt(2))))),case2262=dict(Q=[15,0],D=[6,8],QD_squared=145,QC_squared=64),case548=dict(rejected_equality_both_sides=144,all_three_digit_fixed_points=[145])))
subprocess.run([sys.executable,str(S/'scripts/build_teacher7_expansion.py')],check=True)
build=json.loads((S/'audits/teacher7_expansion_build.json').read_text())
jobs=[dict(name=dataset+'_lr5e5_s43',train_file=str(S/f'data/{dataset}.jsonl'),lr=5e-5,stop=4) for dataset in ['teacher7_expansion_combined','teacher7_expansion_repeat_control','teacher7_expansion_rawnew_control']]
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text())
assert not any(j['name'] in {q['name'] for q in queue} for j in jobs)
idx=next(i for i,j in enumerate(queue) if j['name']=='expansion32_combined_lr1e5_s43')
queue[idx:idx]=jobs
prioritypath=S/'results/dev_priority.json';priority=json.loads(prioritypath.read_text())
idx=priority.index('expansion32_combined_lr1e5_s43_epoch4')
priority[idx:idx]=[j['name']+'_epoch4' for j in jobs]
write(dest,dict(time=time.time(),prospective_plan_sha256=sha(S/'audits/teacher7_expansion_plan.json'),review_decisions_sha256=sha(S/'audits/teacher7_expansion_review_decisions.json'),exact_checks_sha256=sha(S/'audits/teacher7_expansion_review_exact_checks.json'),datasets=build['datasets'],jobs=jobs,queue_total=len(queue),new=build['new'],new_greedy_wrong=build['new_greedy_wrong'],old_core_n=build['old_core_n'],limits=build['control_limitations'],evaluation='Fixed epoch4 seed43 development pilots, horizon8, LR5e-5. Same slots/exposure controls, supervised token counts differ. No OOD outcomes consulted, no multiseed efficacy claim.'))
write(prioritypath,priority);write(queuepath,queue)
print(json.dumps(dict(released=len(jobs),queue_total=len(queue),new=build['new'],hard=build['new_greedy_wrong'],datasets=build['datasets']),indent=2))
