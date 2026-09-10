"""Exact arithmetic checks supporting three process-review exclusions."""
from math import comb
import sympy as sp
from common import *
x=sp.Rational(1,4);y=x
constraint=x*x/y+(1-x)**2/(1-y)
correct_target=(1-y)**2/(1-x)+y*y/x
candidate_wrong_target=(1-x)**2/x+x*x/y
assert constraint==correct_target==1 and candidate_wrong_target==sp.Rational(5,2)
row9=[comb(9,k) for k in range(10)];row10=[comb(10,k) for k in range(11)]
assert 100 not in row9 and 100 not in row10 and comb(100,1)==100
R=sp.sqrt(6)/4;bad_radius=R-sp.sqrt(3)/3
claimed_probability=sp.simplify(5*(bad_radius/R)**3);true_probability=sp.Rational(5,27)
assert abs(float(claimed_probability)-.2)>.19
write(S/'audits/process_counterexamples.json',dict(method='Exact rational/binomial arithmetic and symbolic evaluation, independently checkable; verifies specific intermediate errors, not a population error rate',cases={
 '1441':dict(x=str(x),y=str(y),given_constraint=str(constraint),correct_requested_expression=str(correct_target),candidate_rewritten_expression=str(candidate_wrong_target),explanation='A valid input satisfying the premise makes the candidate rewritten expression 5/2 while the requested expression is 1. The solution only evaluates the special x=y=1/2 case.'),
 '1461':dict(pascal_row9=row9,pascal_row10=row10,correct_witness_row=100,correct_witness_column=1,correct_witness_value=comb(100,1),explanation='The asserted appearances of 100 in row9 or10 are false, although 100 is the correct final answer because row100 contains it.'),
 '208':dict(candidate_radius_over_R=str(sp.simplify(bad_radius/R)),candidate_own_probability_expression=str(claimed_probability),candidate_own_numeric_value=float(claimed_probability),candidate_claimed_numeric_value=.2,correct_probability=str(true_probability),explanation='The candidate geometry is wrong, and even its own radius formula evaluates to about0.00094, not0.2. A correct answer-choice label masks the invalid derivation.')
 }))
print('Three explicit process errors checked with exact/symbolic arithmetic.')
