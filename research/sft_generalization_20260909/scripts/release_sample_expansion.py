"""Completed stratified review, exact checks and student question-scale controls."""
import itertools,math,subprocess
from fractions import Fraction
from collections import Counter
import sympy as sp
from common import *

notes={
2080:('retain','Rectangle diagonal10 is circle diameter; circumference10pi.'),
1118:('retain','Coordinates match Asymptote; shoelace area9.'),
220:('exclude_error','Several explicit false comparisons194/180/170/164<142; invented integer-side search rather than Pythagorean area identity.'),
1204:('exclude_proof_gap','Checks only feasiblex2,y4 without establishing uniqueness. No false equation; final8 independently confirmed. Conservative missing-justification exclusion, not wrong-answer evidence.'),
1985:('retain','Five strips give rectangle perimeter12s/5=36; square perimeter60.'),
2665:('exclude_error','Listed8figures contain only2triangles andwrongorder; theninventsthreetrianglepositions/rows. ActualAsymptotehastrianglespositions2,4,7.'),
1963:('retain','Smallradius1,larger4;difference15pi.'),
844:('exclude_minor','Callsfirstsurroundingringoutermostdespitefurtherouterrings. Arithmetic8n/800correct;conservativeminorspatialwordingexclusion,notincorrectcount.'),
885:('retain','Equalbaseexponentsyieldroots2,3,sum5.'),
2109:('retain','Consecutiveintegeraveragex2,conditionx>0,min1.'),
3213:('retain','Quadraticina=sin²x hasminimum1/2within[0,1].'),
598:('exclude_error','Falseinitialrewritingfixesxto±5andclaimsy²only0or9;valid(3,4)hasy²16. Later12-pointlistcorrectdoesnotrepairfalsepremise.'),
571:('exclude_error','Wrongarraystartingterm/rowratio/columnratioandgeometricseriesarithmetic. Actualsum>1,claimed1/2008;finalmod1coincidental.'),
1328:('retain','Areaequationgivesx4/5bothpositivelengths,sum9.'),
2558:('retain','Squares10201and9801difference400.'),
1538:('retain','Subtractnormalprojection(3/2,-1/2,2),result(-1/2,5/2,1).'),
47:('retain','Gridlength4andshadedsidesqrt2givearea2/16=1/8.'),
3127:('exclude_error','Inventedsquarewithsidex+yanddiagonal9contradictsx²+y²81;wouldrequirexy=-81/4.'),
1704:('exclude_error','FalseAEinterceptedarc2/7circleandexternal/internalangleclaims;actualarc3/7circle. Final540happenstobecorrect.'),
1820:('retain','Counts17golfers52rounds;52/17roundsto3.'),
2740:('retain','Eachsemicirculararcradius1/pihaslength1;fourarcsperimeter4.'),
1707:('retain','Normallineh4k18andequidistance4h6k19yield(-16/5,53/10).'),
2382:('exclude_error','WrongWXYZcoordinatesandincorrectshoelaceevaluation. Listedbowtiepointshavezerosignedarea,actualsquarehasarea1/2.'),
536:('exclude_error','Invented4largest+16+16classificationdoesnotidentifyvaliddisjointclasses;completecoordinateenumerationfindssizes16,10,8,2includingonly2largest.'),
676:('exclude_error','Treatsmod10residueasactualLucasvalue: L10=123,not3. UseswrongindexL3withoutperiodargument;finaldigitcoincidental.'),
1199:('retain','Openingwordingambiguousbutexplicitdefinitionsand3x2y540,x=y-40consistent;D132.'),
1005:('exclude_error','Incorrectrootapproximationsleadstofalseclaimlargestbaseforb8is1;2^8=256<399. Final19+2correct.'),
1046:('retain','Correcttelescopingidentityandtailvanishingforx>1.'),
379:('retain','Binary216decimal;successivedivisionsgiveterms3120base4.'),
2698:('exclude_error','w^5=32doesnothaveroot-2;missesfourcomplexrootsandusesnonrootz=-1/3asdiameterendpoint.'),
314:('retain','Complexmultiplicationgives29i.'),
2460:('retain','Roots-4and1/5,integers-4..0correct. Statedhalf-openintervalhasexactlysamerelevantintegersasclosedcomplement;notanerrorfortheaskedintegerdomain.')}
review=S/'audits/sample_expansion_review.jsonl';rows=read(review);assert len(rows)==32 and set(notes)=={r['expansion_index'] for r in rows}
excluded=sorted(i for i,(d,_) in notes.items() if d.startswith('exclude'))
assert len(excluded)==13
decisions=[dict(expansion_index=r['expansion_index'],target_sha256=hashlib.sha256(r['solution'].encode()).hexdigest(),review_group=r['review_group'],decision=notes[r['expansion_index']][0],reason=notes[r['expansion_index']][1]) for r in rows]
write(S/'audits/sample_expansion_review_decisions.json',dict(time=time.time(),review_sha256=sha(review),reviewer='AI research assistant, not independent human expert',decisions=decisions,excluded_expansion_indices=excluded,categories=dict(Counter(r['decision'] for r in decisions)),limitation='Equal8-per-stratum review oversamples diagram problems; not overall error-rate estimate.11substantiveerrors,1proofgap,1minorwordingexclusion are distinguished.'))

# Exact coordinate triangle enumeration. Coordinates scaled byx/7.5,y/10.
P=sp.Point;segments=[sp.Segment(P(0,y),P(4,y)) for y in [0,1,2]]+[sp.Segment(P(x,0),P(x,2)) for x in range(5)]+[sp.Segment(P(*a),P(*b)) for a,b in [((0,0),(2,2)),((2,0),(0,2)),((2,0),(4,2)),((4,0),(2,2))]]
vertices={p for line in segments for p in line.points}
for a,b in itertools.combinations(segments,2):
 for p in a.intersection(b):
  if isinstance(p,sp.Point):vertices.add(p)
triangles=[]
for pts in itertools.combinations(sorted(vertices,key=lambda p:(p.x,p.y)),3):
 area=abs(sp.Polygon(*pts).area) if not sp.Point.is_collinear(*pts) else 0
 if area and all(any(a in line and b in line for line in segments) for a,b in itertools.combinations(pts,2)):
  triangles.append(dict(vertices=[[str(p.x),str(p.y)] for p in pts],scaled_area=str(area)))
counts=dict(Counter(t['scaled_area'] for t in triangles));assert len(triangles)==36 and counts=={'1/2':16,'1':10,'2':8,'4':2}
lucas=[2,1]
for _ in range(122):lucas.append(sum(lucas[-2:]))
assert lucas[10]==123 and lucas[123]%10==4
p=2008;array_sum=Fraction(2*p*p,(2*p-1)*(p-1));assert array_sum>1 and (array_sum.numerator+array_sum.denominator)%p==1
feasible=[(x,y) for x in range(2,4) for y in range(2,5) if x*y+4*x+3*y==28];assert feasible==[(2,4)]
wrong=[P(1,1),P(1,0),P(2,1),P(2,0)];actual=[P(sp.Rational(3,2),sp.Rational(3,2)),P(1,1),P(sp.Rational(3,2),sp.Rational(1,2)),P(2,1)]
def shoelace(points):return abs(sum(a.x*b.y-a.y*b.x for a,b in zip(points,points[1:]+points[:1])))/2
z=-sp.Rational(1,3);residual=sp.simplify((z+1)**5-32*z**5);assert residual==sp.Rational(64,243)
write(S/'audits/sample_expansion_review_exact_checks.json',dict(time=time.time(),case220=dict(claimed_less_than142=[194,180,170,164],all_actually_greater=True),case1204=dict(bound='x,y>1 andxy4x3y28implyx<=3,y<=4',all_feasible=feasible),case2665=dict(actual_figure_sequence=['sq','tri','circ','tri','sq','circ','tri','sq'],actual_triangle_positions=[2,4,7],target_list_triangle_positions=[3,6]),case598=dict(valid_counterexample=[3,4],y_squared=16),case571=dict(correct_sum=str(array_sum),claimed_sum='1/2008',correct_remainder=1),case3127=dict(implied_xy='-81/4',positive_xy_required=True),case1704=dict(actual_intercepted_arc_degrees='1080/7',claimed_arc_degrees='720/7',actual_tip='540/7'),case2382=dict(wrong_coordinate_shoelace=str(shoelace(wrong)),actual_area=str(shoelace(actual))),case536=dict(vertex_count=len(vertices),triangle_count=len(triangles),scaled_area_counts=counts,triangles=triangles),case676=dict(L10=lucas[10],L123_last_digit=lucas[123]%10),case1005=dict(counterexample_base=2,exponent=8,power=256,less_than399=True),case2698=dict(nonroot='-1/3',polynomial_residual=str(residual),raw_reference_note='Rawreferenceexpandedcirclesign+1isalsoatypo;correctmoduluslocus3x²-2x+3y²-1=0givesradius2/3.')))
subprocess.run([sys.executable,str(S/'scripts/build_sample_expansion.py')],check=True)
build=json.loads((S/'audits/sample_expansion_build.json').read_text());assert build['new']==2588 and build['datasets']['sample_expansion_combined']['n']==4149
jobs=[]
for lr,label in [(1e-5,'1e5'),(5e-5,'5e5')]:
 for dataset in ['sample_expansion_combined','sample_expansion_repeat_control']:
  jobs.append(dict(name=dataset+f'_lr{label}_s43',train_file=str(S/f'data/{dataset}.jsonl'),lr=lr,stop=4))
for dataset in ['sample_expansion_rawnew_control','sample_expansion_common_teacher32','teacher32_expansion_common_sample']:
 jobs.append(dict(name=dataset+'_lr1e5_s43',train_file=str(S/f'data/{dataset}.jsonl'),lr=1e-5,stop=4))
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text());assert not any(j['name'] in {q['name'] for q in queue} for j in jobs)
idx=next(i for i,j in enumerate(queue) if j['name']=='expansion32_combined_lr1e5_s43')
queue[idx:idx]=jobs
write(S/'audits/sample_expansion_training_release.json',dict(time=time.time(),review_decisions_sha256=sha(S/'audits/sample_expansion_review_decisions.json'),exact_checks_sha256=sha(S/'audits/sample_expansion_review_exact_checks.json'),datasets=build['datasets'],jobs=jobs,queue_total=len(queue),new=build['new'],new_greedy_wrong=build['new_greedy_wrong'],old_core_n=build['old_core_n'],limits=build['control_limitations'],evaluation='Development pilots; no OOD outcomes consulted, no multi-seed efficacy claim.'))
write(queuepath,queue)
print(json.dumps(dict(released=len(jobs),queue_total=len(queue),new=build['new'],hard=build['new_greedy_wrong'],datasets=build['datasets'])))
