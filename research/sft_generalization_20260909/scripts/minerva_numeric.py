"""Finite real numeric grading for OCW-style answers, without absolute rounding floors."""
import math,re,signal
from decimal import Decimal
from sympy import Basic,Equality,Symbol
from math_verify import parse
from evaluation.audit_lora_sft_review import boxed_contents
LITERAL=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
SCIENTIFIC=re.compile(r'(?<![\w\\.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE]([+-]?\d+)(?![\w.])')
SPECIAL={'np.arcsin(10/13)':math.asin(10/13),'-1./3':-1/3,'-3./2':-3/2}
def gold_numeric(answer):
 s=answer.strip()
 if LITERAL.fullmatch(s):
  d=Decimal(s);v=float(d);assert math.isfinite(v) and (v!=0 or d==0);return v
 return SPECIAL.get(s)
def normalize_scientific(s):
 return SCIENTIFIC.sub(lambda m:'('+m.group(1)+r'\times10^{'+str(int(m.group(2)))+'})',s)
def _timeout(signum,frame):raise TimeoutError('Numeric expression evaluation exceeded2seconds')
def numeric_prediction(text):
 boxes=boxed_contents(text)
 if not boxes:return dict(value=None,reason='no_complete_box')
 value=boxes[-1].strip()
 if len(value)>2000:return dict(value=None,reason='numeric_box_over2000chars')
 if LITERAL.fullmatch(value):
  try:
   d=Decimal(value);v=float(d)
   if v==0 and d!=0:return dict(value=None,reason='nonzero_literal_underflow')
   return dict(value=v if math.isfinite(v) else None,reason='literal' if math.isfinite(v) else 'nonfinite')
  except Exception:return dict(value=None,reason='literal_parse_failed')
 normalized=normalize_scientific(value)
 try:expressions=parse(r'\boxed{'+normalized+'}')
 except Exception as e:return dict(value=None,reason='parse_exception',detail=type(e).__name__)
 candidates=[e for e in expressions if isinstance(e,Basic)]
 if not candidates:return dict(value=None,reason='no_symbolic_parse')
 expr=candidates[0]
 if isinstance(expr,Equality) and isinstance(expr.lhs,Symbol):expr=expr.rhs
 if not getattr(expr,'is_number',False) or expr.free_symbols:return dict(value=None,reason='not_a_constant_number')
 old_handler=signal.signal(signal.SIGALRM,_timeout);old_timer=signal.setitimer(signal.ITIMER_REAL,2)
 try:
  evaluated=expr.evalf(25);v=float(evaluated)
  if v==0 and evaluated.is_zero is not True:return dict(value=None,reason='nonzero_expression_underflow')
  return dict(value=v if math.isfinite(v) else None,reason='expression' if math.isfinite(v) else 'nonfinite')
 except TimeoutError:return dict(value=None,reason='numeric_evaluation_timeout')
 except Exception as e:return dict(value=None,reason='not_finite_real',detail=type(e).__name__)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
  if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*old_timer)
def numeric_equal(prediction,reference,rtol=1e-4):
 return prediction is not None and math.isfinite(prediction) and math.isclose(prediction,reference,rel_tol=rtol,abs_tol=0.)
