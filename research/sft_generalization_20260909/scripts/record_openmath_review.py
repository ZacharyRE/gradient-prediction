from common import *
rr=read(S/'audits/openmath_target_review.jsonl')
reasons=[
'Under usual interpretation Alex plus120 arrivals minus30 departures gives91; wording about all staying and unspecified friends is ambiguous, retained with note.',
'Correct4x=480, meetings120, assuming described activities exhaust work time.',
'Correct180-50-20=110.',
'Exact45% of175 is78.75 discrete messages; arbitrary rounding down unsupported. Synthetic problem/expected97 not well-defined as stated.',
'Correct104/2*4.50=234.',
'Correct42+(42+9)=93.',
'Correct(32-8)/240*100=10 liters per100km.',
'Correct revenue120+30 and150/5=30.',
'False maximum: feasible(2/5,2/5,6/5) gives44/25>4/3. Missing cross term and wrong constraint algebra; gold majority label itself false.',
'Angle-bisector theorem applied to non-opposite AD instead of AC, EC replaced by ED, and equal altitudes from B/C to AD unsupported.',
'Underspecified operation; convergence does not imply sum of squared increments zero. f(x)=5 gives valid sequence2,5,5,... with sum9.',
'For a1, f(x)=x-1 where defined, f(f(3))=1 not9. Lost degree5 term in cross multiplication and incompatible coefficient equations declared jointly satisfied.',
'Reflecting one vertex does not make arbitrary pentagon regular; apothem confused with circumradius; multiple invalid radical simplifications and unsupported final minimum.',
'At least3 per color does not force4 each. Gives no actual count before36. Exact enumeration gives3372 labeled-chair assignments (mod1000=372), or286 rotational classes, both different.',
'Correct linear system r26/7,s29/7, product754/49.',
'Wrong forbidden-distance interpretation and invented disjointness; independent path counts give maximal A1003,B1002,total2005, not383.',
'Correct remainder45% of300=135.',
'Correct22.5n=30k, n4/8/12/16.',
'Four drawn white triangles are disjoint except boundaries and each1/8 area, giving225. Candidate swaps shaded/unshaded labels; both halves equal, retained with explicit wording caveat.',
'False trig identity: at A0 expression is0, not1; multiplying expression by itself is not a valid cancellation.',
'Law of sines does not imply right triangle. Equilateral triangle satisfies supplied side/sine condition and has largest angle60, contradicting90.',
'Correct first7 nights then eighth-day escape.',
'Cube root28 is not3; b2 gives h(j(b))=26 not27. Correct b=(1+cuberoot28)/2.',
'Five musicians does not specify number of available songs or one song per musician; assumes unstated5-song repertoire.',
'Correct180/1.5/12=10 feet.',
'Correct3000/.15=20000 and61% remaining12200.',
'Correct total tools6vs11 difference5.',
'Correct5*8+2*24=88.',
'Correct24-8-4+24=36.',
'Correct(18*2+10*4)*5*2000=760000.',
'Correct usual weekly rate168/6=28; endpoint dosing convention implicit in word problem.',
'Correct per-cake revenue15 minus ingredients6 and packaging1 gives8.'
]
bad={3,8,9,10,11,12,13,15,19,20,22,23};minor={0,18,30};assert len(rr)==len(reasons)==32
records=[dict(review_index=i,problem_sha256=r['problem_sha256'],review_group=r['review_group'],decision='exclude' if i in bad else 'retain_with_wording_or_convention_note' if i in minor else 'retain',reason=reasons[i]) for i,r in enumerate(rr)]
write(S/'audits/openmath_target_review_decisions.json',dict(time=time.time(),records=records,excluded_problem_hashes=[r['problem_sha256'] for r in records if r['decision']=='exclude'],status='Entire external dataset quarantined pending independent weak process audit and further calibration; no training release',reviewer='AI research assistant; exact counterexamples are reproducible, not independent human gold',limitation='32 stratified cases,8 eachsource/length group. Not overall source prevalence; public dataset quality is not thereby the cause of earlier localSFT results.'))
print('Reviewed32, excluded',len(bad),'; entire dataset remains quarantined')
