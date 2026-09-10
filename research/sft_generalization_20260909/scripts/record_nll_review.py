"""Record assistant review of the frozen 48-target stratified sample."""
from common import *
rows=read(S/'audits/nll_noguided_review.jsonl')
reasons=[
'Correct base conversion gives 12+d=11d+2, d=1.',
'Correct repeated root from (x+3)^2=0.',
'Correct Vieta denominator8 and numerator364, result91/2.',
'Correct completing squares, center(2,6).',
'Correct trapezoid area/base ratio7/3, AB147.',
'Correct logarithm exponent7/2.',
'Correct inequality x<13/5 and greatest integer2.',
'Correct probability product21/110.',
'Correct360 total and120 ending5; divisibility by3 follows from digit sum24, omitted routine check.',
'Excludes valid(3,6) from its final list after computing it, contradicting its own enumeration before announcing9 stamps.',
'Correct normal(6,6,-6) and plane x+y-z+1=0.',
'Correct monetary rounding to65.13.',
'False claim w(x-y) is maximized at x=y; that term is then zero. Unjustified global restriction and later dropped term invalidate maximum proof.',
'Correct square-divisor exponent choices2*5=10.',
'Invented h(4)=2 contradicts h(4)=6; invents h(12), transforms points incorrectly and claims absent point is listed.',
'Correct102km over2h gives51.',
'False ordered residue-class count: not every0mod4 integer precedes every2mod4 integer. Counts625 for that orientation instead of300 and omits reverse orientations; cancellations accidentally preserve1850.',
'False table entries: two-digit multiples of23 include92 and of24 include96, each four multiples; stated largest69/72 and counts3 are false.',
'Mass assignments contradict midpoint and angle-bisector constraints; claimed LP/PC and2/5*AM are unsupported. Correct final72 does not validate derivation.',
'Correct angle bisector ratio, AX50/3.',
'Correct polygon vertices and shoelace sums74,102, area14.',
'Correct alternating-digit counts72+108=180.',
'Correct offset-box volume60+94+12pi+4pi/3 and coefficient sum505.',
'Correct cyclic geometry and area ratio(2+sqrt3)/(2pi), sum7.',
'Correct price equations give corn13.5.',
'Correct six-term geometric sum63/64.',
'Correct distance roots11,-1 and positive-root selection.',
'Correct absolute-value equation x=3/2; absolute value of negative1/2 equals positive1/2.',
'Correct angle reduction tan1000=tan(-80).',
'Correct C(8,3)=56.',
'Correct remainder100 mod12=4.',
'Correct complex arithmetic18+10i.',
'Correct abcba representation and9*10*10=900; calls middle c fourth digit instead of third, a minor ordinal wording error with unchanged counting.',
'Figure1 has8 sides, not4; perimeter17 and the claimed segment removals are unsupported, followed by correct19 via invalid derivation.',
'Correct inclusion-exclusion74/100=37/50.',
'Correct squares force(5,-3), sum2.',
'After fixing terminal5, remaining digit sum is19 not24; computes probability1 then jumps to boxed1/3 without correction.',
'Correct quadratic minimization in t then x^2/2+1/x^2, unique positive critical root x^4=2; globality implicit from convexity/endpoints.',
'Correct reroll recurrence E=1+E/6 and365E=438.',
'Correct toothpick progression4+249*3=751.',
'False modular-division identity: P=127512000 gives(P/1000)mod10=2 but((P mod1000)/1000)mod10=0; later full product computation does not repair false rule.',
'Testing a few partitions and finding34 twice does not exclude sum4 per group; essential infeasibility/global-bound argument missing.',
'False general claim that7*11*13 is the only factorization into three positive integers;1*7*143 is a counterexample. Required pair sums>=2 are not stated in uniqueness argument.',
'Pointwise square equality does not imply a single global sign branch; moreover substitutions for f(t)=-t and -t-1 give wrong left sides. Correct final6 but invalid elimination.',
'Correct arithmetic-series equation x(x+1)/2=4x gives7 rounds and35 coins.',
'Correct derivative, critical x9, endpoints and11. Strict concavity of the sum of square roots supplies implicit globality, analogous to retained one-variable convexity omissions.',
'Correct factorization and inequality x/(x-2)<=0; square zero at1/5 already mentioned and lies inside returned[0,2). Sign wording loose but negative accounted for.',
'Uses schematic Asymptote metric AB=radius although stated AB=6 and radius=sqrt50; internally contradicts B=(5,-1). Reject metric use of schematic drawing, despite correct final26.'
]
assert len(reasons)==len(rows)==48
bad={9,12,14,16,17,18,33,36,40,41,42,43,47}
minor={8,32,37,45,46}
records=[dict(review_index=i,original_index=r['original_index'],target_sha256=r['target_sha256'],review_dataset=r['review_dataset'],review_group=r['review_group'],decision='exclude' if i in bad else 'retain_with_noted_omission_or_wording' if i in minor else 'retain',reason=reasons[i]) for i,r in enumerate(rows)]
write(S/'audits/nll_noguided_review_decisions.json',dict(time=time.time(),reviewer='AI research assistant, not independent human expert',review_sha256=sha(S/'audits/nll_noguided_review.jsonl'),policy='Exclude false mathematical statements, contradictory reasoning and essential proof gaps. Retain harmless ordinal wording and routine convexity/algebra omissions with explicit notes. This stratified sample is not an overall error-rate estimate or whole-corpus certificate.',records=records,excluded_target_keys=[[r['original_index'],r['target_sha256']] for r in records if r['decision']=='exclude']))
