"""Semantic numeric-reference audit BEFORE any Minerva model outputs."""
from common import *
from minerva_numeric import *
from math_verify import verify
assert not list((S/'results/evaluation/ood_minerva').glob('*/math_predictions.jsonl'))
rows=read(S/'data/ood_minerva.jsonl');numeric=[];checks=[];scientific=[]
for i,row in enumerate(rows):
 g=gold_numeric(row['answer'])
 if g is None:continue
 literal=format(g,'.17g');positive=numeric_prediction(r'\boxed{'+literal+'}')
 assert numeric_equal(positive['value'],g)
 bad=2*g if g!=0 else 1.;negative=numeric_prediction(r'\boxed{'+format(bad,'.17g')+'}')
 assert not numeric_equal(negative['value'],g)
 if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+',row['answer'].strip()):scientific.append(i)
 numeric.append(dict(index=i,answer=row['answer'],numeric_value=g,positive=literal,negative=format(bad,'.17g'),positive_passed=True,negative_rejected=True))
assert len(numeric)==191 and len(scientific)==58
for text,g,expected in [(r'\boxed{4.5e33}',4.5e33,True),(r'\boxed{4.5\times10^{33}\ \mathrm{erg/s}}',4.5e33,True),(r'\boxed{x=4.5\times10^{33}}',4.5e33,True),(r'\boxed{1e-19}',1e-19,True),(r'\boxed{2\times10^{-19}}',1e-19,False),(r'\boxed{0}',1e-19,False),(r'\boxed{1\times10^{-19}}',0.,False),(r'\boxed{-1/3}',-1/3,True),(r'\boxed{\arcsin(10/13)}',math.asin(10/13),True),(r'\boxed{1.6\text{ cm}}',1.6,True),(r'\boxed{x+1}',1.,False),(r'\boxed{1+2i}',1.,False),(r'\boxed{1e-400}',0.,False),(r'\boxed{10^{-400}}',0.,False),('No final answer',1.,False)]:
 p=numeric_prediction(text);correct=numeric_equal(p['value'],g);assert correct==expected,(text,p,g)
 checks.append(dict(prediction=text,reference=g,expected=expected,actual=correct,parse=p))
legacy=[dict(reference=r'\boxed{4.5e33}',prediction=r'\boxed{4.5\times10^{33}}',legacy_correct=bool(verify(parse(r'\boxed{4.5e33}'),parse(r'\boxed{4.5\times10^{33}}')))),dict(reference=r'\boxed{1\times10^{-19}}',prediction=r'\boxed{2\times10^{-19}}',legacy_correct=bool(verify(parse(r'\boxed{1\times10^{-19}}'),parse(r'\boxed{2\times10^{-19}}'))))]
assert not legacy[0]['legacy_correct'] and legacy[1]['legacy_correct']
write(S/'audits/minerva_numeric_contract.json',dict(time=time.time(),script_sha256=sha(__file__),helper_sha256=sha(S/'scripts/minerva_numeric.py'),dataset_sha256=sha(S/'data/ood_minerva.jsonl'),passed=True,n_numeric=191,n_symbolic=81,n_scientific_literals=58,numeric_references=numeric,unit_and_representation_checks=checks,legacy_counterexamples=legacy,primary_rtol=1e-4,primary_atol=0.,sensitivity_rtols=[.01,.05],scope='Before ANYOOD model outputs. Numeric grading uses lastcompleteboxed, safe scientific notation, finite real evaluation andrelative1e-4withzeroabsolute tolerance. Both original/strict analytic vectors use this numeric score;81symbolic retain respective original/strict scoring. Legacy savedmodel scores remain immutable. Sensitivity1%/5% are tolerance diagnostics, not automaticgroundtruthcertification or posthocsuccesscriteria. Not a replication of all originalOCW extraction/unit/nearzero conventions.'))
print(json.dumps(dict(passed=True,n_numeric=191,n_symbolic=81,n_scientific=58,n_synthetic_checks=len(checks))))
