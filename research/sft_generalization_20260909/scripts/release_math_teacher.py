"""Record completed assistant process review and freeze math-teacher controls."""
import subprocess
import sympy as sp
from common import *
review=S/'audits/math_teacher_review.jsonl';rows=read(review)
notes={
1034:'Squared ratio is nonnegative whenever denominator nonzero; excludes3 correctly.',
161:'Six gcd/LCM computations correct; maximum60.',
1581:'6-8-10 right triangle area24, rectangle4x6 perimeter20.',
853:'Midpoint(3,6), perpendicular slope-1, b9.',
542:'Substitution x=-1/(z-1),y=(z-1)/z gives xyz=-1; excluded denominators follow original system.',
999:'Tangent discriminant1600-32k=0 gives50.',
307:'Conjugate sum/product real; p=-5,q²2, a=b=0.',
760:'Continuity roots3,-1 sum2; target correct despite a sign typo in raw reference final Vieta expression.',
922:'Factorial ratio exactly2006+1/(2005*2007); floor2006. Wording cancellation is loose but displayed algebra valid.',
652:'Legendre valuation16! at2 is15;2^15 ends8.',
1243:'Uniform team then pair gives(1/10+1/21+1/28)/3=11/180.',
351:'Real cube-root substitution y=1±sqrt3; smaller cube10-6sqrt3, p+q118.',
1703:'Positive-base square root simplification yields18.',
1427:'Log substitution sum2b/product2b4, discriminant and b>0 give(0,1/sqrt2]; real logarithms yield positive x,y.',
92:'Cubic factors correct; explicitly removes extraneous denominator root3, remaining-1,-6 difference5.',
1203:'Positive exponential square root gives6*3^t=18, t1.',
739:'Prime-exponent dominance argument compressed but valid; b12,n15 maximizes divisor count496. Complete225-pair check stored.',
485:'Exclude: for product(x+p), coefficients a and c are positive elementary sums, not negative. Later f(1) statement also subtracts1 incorrectly; correct final528 does not repair false algebra.',
493:'All digit sums24, last digit5; multiset arrangements120/360=1/3.',
538:'Triple tangent47/52, subsequent992/993, yieldsx1985; angles here remain in principal branches.',
887:'Exclude: confuses solving f(x)=f(-x) with identity f(x)=f(4-x); claims intersection atx2, actualx0. f(2)=-1/2 butf(-2)=15/2.',
151:'Five equally likely red faces, four onRRcards, conditional4/5.',
421:'n=15k²,positivek1..8 gives8.',
213:'Summation shift and arithmetic valid, S4. Convergence omitted but follows two-step maximum contraction7/12; independently checked characteristic rootsabs<1.'}
assert set(notes)=={r['original_index'] for r in rows} and len(rows)==24
decisions=[dict(original_index=r['original_index'],target_sha256=hashlib.sha256(r['solution'].encode()).hexdigest(),review_group=r['review_group'],decision='exclude' if r['original_index'] in [485,887] else 'retain',reason=notes[r['original_index']]) for r in rows]
write(S/'audits/math_teacher_review_decisions.json',dict(time=time.time(),review_sha256=sha(review),reviewer='AI research assistant; not independent human expert',decisions=decisions,excluded_target_indices=[485,887],limitation='Preselected stratified24-case review, not representative overall error-rate estimate or whole-pool certification.'))
x=sp.symbols('x');f=(x-1)*(x-3)/2;poly=sp.expand((x+1)*(x+2)*(x+4)*(x+66))
values=[(b,n,int(sp.prod(e*n+1 for e in sp.factorint(b).values()))) for b in range(1,16) for n in range(1,16)]
best=max(v[2] for v in values);assert best==496
assert sp.simplify(f-f.subs(x,-x))==-4*x and f.subs(x,2)!=f.subs(x,-2)
assert poly.coeff(x,3)==73 and poly.coeff(x,1)==932
roots=sp.solve(x*x-x/3-sp.Rational(1,4),x);assert all(abs(float(r))<1 for r in roots)
write(S/'audits/math_teacher_review_exact_checks.json',dict(time=time.time(),case485=dict(polynomial=str(poly),a=int(poly.coeff(x,3)),c=int(poly.coeff(x,1)),f1=int(poly.subs(x,1)),claimed_a=-73,claimed_c=-932),case887=dict(f_minus_reflection=str(sp.expand(f-f.subs(x,-x))),actual_intersection_x=[0],claimed_x=2,f2=str(f.subs(x,2)),f_minus2=str(f.subs(x,-2))),case739=dict(exhaustive_b_n_pairs=len(values),maximum=best,achievers=[v for v in values if v[2]==best]),case213=dict(characteristic_roots=[str(r) for r in roots],both_abs_less_than1=True)))
subprocess.run([sys.executable,str(S/'scripts/build_math_teacher_targets.py')],check=True)
build=json.loads((S/'audits/math_teacher_build.json').read_text());assert build['accepted']==1607
jobs=[]
for lr,label in [(1e-5,'1e5'),(5e-5,'5e5')]:
 jobs.append(dict(name=f'teacher_math7_all_lr{label}_s43',train_file=str(S/'data/teacher_math7_all.jsonl'),lr=lr,stop=4))
for data in ['teacher_math7_common_teacher7','teacher7_common_math7']:
 jobs.append(dict(name=data+'_lr1e5_s43',train_file=str(S/f'data/{data}.jsonl'),lr=1e-5,stop=4))
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text());names={j['name'] for j in queue}
assert not any(j['name'] in names for j in jobs)
# The already running serial replay is left untouched. Release pilots next.
idx=next(i for i,j in enumerate(queue) if j['name']=='sample_augmented_lr1e5_s43_serial_replay')+1
queue[idx:idx]=jobs[:2]
idx=next(i for i,j in enumerate(queue) if j['name']=='sample_all_all64_lr1e5_s43')
queue[idx:idx]=jobs[2:]
write(S/'audits/math_teacher_training_release.json',dict(time=time.time(),review_decisions_sha256=sha(S/'audits/math_teacher_review_decisions.json'),datasets=build['datasets'],accepted=build['accepted'],historical_greedy_wrong=build['historical_greedy_wrong'],jobs=jobs,queue_total=len(queue),interpretation='Teacher checkpoint plus official CoT prompting changes target pipeline. Student and student prompts unchanged. Same-question general7 control isolates source pipeline from coverage, not teacher weights from prompt or target length. Development pilots, not multi-seed efficacy proof.'))
write(queuepath,queue)
print(json.dumps(dict(released=len(jobs),queue_total=len(queue),datasets=build['datasets'])))
