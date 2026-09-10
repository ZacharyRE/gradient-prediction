"""Record eight selected heldout transition reviews with elementary exact checks."""
from fractions import Fraction
from itertools import permutations
import sympy as sp
from common import *

rows=read(S/'audits/consistent_transition_review.jsonl')
notes={
1177:dict(baseline='Incorrectly chooses which three dresses survive and counts5! assignments; ruined dresses are already fixed.',adapted='All three use5P3=60, a valid assignment count. Wording about order is imprecise but count is assignments to the three distinct fixed outfits, not an extra runway-order factor.',classification='answer_gain_with_valid_core'),
543:dict(baseline='Lists60 in both lists then omits60 from their intersection and returns9.',adapted='All three use divisibility and list allten multiples6..60; these lie within the twenty multiples3..60.',classification='answer_gain_with_valid_core'),
1499:dict(baseline='Correct proportionality42/9=x/6 gives28.',adapted='Seeds43/45 falselysimplify9/42=1/4, also write x/x=1/4, thenreturn24. Seed44attributesnormal6toLynn(was9) and reverses conversion, returning6/7.',classification='answer_loss_arithmetic_and_entity_binding'),
893:dict(baseline='Correct finalfactorization9951=3*31*107. Several peripheral decimal division estimates are inaccurate, so do not certify every baseline sentence; exactfactorization core is valid.',adapted='All three falselydeclare3317 notdivisible31. Seeds43/45 invent3317=67*49 andreturn67; seed44invents3317=37*89 andreturns89.',classification='answer_loss_false_division_and_factorization'),
459:dict(baseline='Startswith2asdivisible4 and usesstep4 while requiringunits2; neithersequence norcount49 valid.',adapted='All three correctly list finalten numbers andreturn10, but intermediate bullets falselysay132and172notdivisible4, contradicting final list. Answer improves; process is not clean.',classification='answer_gain_with_internal_process_errors'),
772:dict(baseline='Incorrect composition ofexponents and invalid argumentreturn13.',adapted='All three repairinitialcompositionz143=z androotorder142; finalcoprimeness argumentsstillinvalid: seed43n71k,mkdoesnotgenerallysolvek/71=m/n; seed44k1,m2,n71violateskn=71m; seed45falselyrestrictsk1. Equal sines also admit supplementary angles. Correct denominator71 follows instead frompositiveimaginarypart,k1..70,andprimality71.',classification='answer_gain_with_invalid_final_derivation'),
766:dict(baseline='Correctanglebisector+PythagorasgivesCD5,BC9,AC15.',adapted='Seed43usesBCas hypotenuse despite rightangleB, thenmakescontradictory4=12stepandreturns16. Seeds44/45applyPythagorastotriangleACD without arightangle andreturn12.',classification='answer_loss_wrong_geometric_roles'),
1030:dict(baseline='Correct f(-3)=-1 then f(-1)=1 givesg(f(-3))=1.',adapted='All three correctlyfindf(-3)=-1 thenclaimthatfirststepalreadyestablishedf(-1)=-1, returning-1.',classification='answer_loss_function_argument_binding')}
assert set(notes)=={r['index'] for r in rows}
assert len(list(permutations(range(5),3)))==60
intersection=sorted(set(range(3,61,3))&set(range(6,61,6)));assert len(intersection)==10
assert Fraction(9,42)==Fraction(3,14)!=Fraction(1,4) and Fraction(42*6,9)==28
factors=sp.factorint(9951);assert factors=={3:1,31:1,107:1}
assert 3*67*49!=9951 and 3*37*89!=9951 and 3317//31==107 and 3317%31==0
units=[n for n in range(1,201) if n%10==2 and n%4==0];assert len(units)==10 and 132 in units and 172 in units
assert sp.isprime(71) and all(sp.gcd(k,71)==1 for k in range(1,71))
assert Fraction(2,71)!=Fraction(2,142) and 1*71!=71*2 and 2*71==71*2
assert 12**2+9**2==15**2 and Fraction(4,5)==Fraction(12,15) and 15**2+5**2!=12**2+4**2
assert (-3+2)==-1 and (-1+2)==1 and Fraction(1,(-3+2)+2)==1
records=[dict(index=r['index'],group=r['group'],sample_hash=r['sample_hash'],**notes[r['index']]) for r in rows]
write(S/'audits/consistent_transition_review_decisions.json',dict(time=time.time(),review_sha256=sha(S/'audits/consistent_transition_review.jsonl'),records=records,exact_checks=dict(outfit_assignments=60,multiples_intersection=intersection,shoe_ratio=str(Fraction(9,42)),shoe_answer=28,prime_factors={str(k):v for k,v in factors.items()},wrong_factor_products=[3*67*49,3*37*89],units2_divisible4=units,root_unity_denominator=71,triangle_AB_BC_AC=[12,9,15],function_composition=1),reviewer='AI research assistant with reproducible exact arithmetic; not independent human gold.',interpretation='Eight deliberately selected normal-stop, three-seed-consistent transitions on reused1500. Demonstrates real answer gains/losses and the distinction between final answers and valid derivations. Does not estimate error prevalence, uniquely identify parameter mechanisms, or turn these examples into independent general efficacy evidence.'))
print(json.dumps(records,indent=2))
