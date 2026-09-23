"""Exact Countdown solver and safe expression verifier (no eval execution)."""
import ast,re
from fractions import Fraction
from collections import Counter

def puzzle_key(r):return (int(r['target']),tuple(sorted(map(int,r['nums']))))
def number_key(r):return tuple(sorted(map(int,r['nums'])))

def solve(nums,target):
 target=Fraction(target);failed=set()
 def dfs(items):
  state=tuple(sorted(x[0] for x in items))
  if state in failed:return None
  if len(items)==1:return items[0][1] if items[0][0]==target else None
  for i in range(len(items)):
   for j in range(i+1,len(items)):
    a,at=items[i];b,bt=items[j];rest=[items[k] for k in range(len(items)) if k not in (i,j)]
    choices=[(a+b,('+',at,bt)),(a*b,('*',at,bt)),(a-b,('-',at,bt)),(b-a,('-',bt,at))]
    if b:choices.append((a/b,('/',at,bt)))
    if a:choices.append((b/a,('/',bt,at)))
    seen=set()
    for val,tree in choices:
     if val in seen:continue
     seen.add(val);ans=dfs(rest+[(val,tree)])
     if ans is not None:return ans
  failed.add(state);return None
 return dfs([(Fraction(n),int(n)) for n in nums])

def expression(tree):
 if isinstance(tree,int):return str(tree)
 op,a,b=tree;return '('+expression(a)+' '+op+' '+expression(b)+')'

def evaluate_tree(tree,steps):
 if isinstance(tree,int):return Fraction(tree)
 op,a,b=tree;x=evaluate_tree(a,steps);y=evaluate_tree(b,steps)
 z={'+':lambda:x+y,'-':lambda:x-y,'*':lambda:x*y,'/':lambda:x/y}[op]()
 steps.append(f'{x} {op} {y} = {z}.');return z

def solution(tree,nums,target):
 steps=[];assert evaluate_tree(tree,steps)==Fraction(target)
 return 'Combine the supplied numbers step by step:\n'+'\n'.join(steps)+f'\nThe expression uses every supplied number exactly once and equals {target}.\n\\boxed{{{expression(tree)}}}'

def problem(nums,target):
 return (f'Countdown puzzle: use the numbers {nums} to make {target}. '
 'Use each supplied number exactly once. You may use only addition (+), subtraction (-), multiplication (*), division (/), and parentheses. '
 'Do not introduce any other numbers into the final expression. Show your calculation, then put the final arithmetic expression (not just its value) inside \\boxed{...}. Use ASCII operators in that expression.')

def extract(text):
 pos=text.rfind('\\boxed{')
 if pos>=0:
  depth=1;j=pos+7
  while j<len(text) and depth:depth+=(text[j]=='{')-(text[j]=='}');j+=1
  return (text[pos+7:j-1],True) if depth==0 else ('',True)
 tags=re.findall(r'<answer>(.*?)</answer>',text,re.S)
 if tags:return tags[-1],False
 # Format-tolerant fallback: last line containing a full arithmetic expression.
 for line in reversed(text.splitlines()):
  line=line.strip().strip('`$ ')
  if re.fullmatch(r'[\d\s()+*/.\-=]+',line) and re.search(r'\d.*[+*/-].*\d',line):return line,False
 return '',False

def normalize(s):
 s=s.replace('\\left','').replace('\\right','').replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('×','*').replace('÷','/').replace('−','-')
 pattern=r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}'
 for _ in range(10):
  new=re.sub(pattern,r'((\1)/(\2))',s)
  if new==s:break
  s=new
 return s.split('=')[0].strip().replace('{','(').replace('}',')').strip('$ ')

def check(text,nums,target):
 ex,boxed=extract(text);ex=normalize(ex);leaves=[]
 def ev(n):
  if isinstance(n,ast.Expression):return ev(n.body)
  if isinstance(n,ast.Constant) and type(n.value) in (int,float):
   v=Fraction(str(n.value))
   if v.denominator!=1:raise ValueError('Non-supplied constant')
   leaves.append(int(v));return v
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
   v=ev(n.operand);return v if isinstance(n.op,ast.UAdd) else -v
  if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
   x,y=ev(n.left),ev(n.right)
   if isinstance(n.op,ast.Add):return x+y
   if isinstance(n.op,ast.Sub):return x-y
   if isinstance(n.op,ast.Mult):return x*y
   return x/y
  raise ValueError('Disallowed expression syntax')
 try:
  value=ev(ast.parse(ex,mode='eval'));inventory=Counter(leaves)==Counter(map(int,nums));correct=inventory and value==Fraction(target)
  return dict(correct=bool(correct),has_box=boxed,expression=ex,valid_expression=True,inventory_valid=inventory,value=str(value))
 except (ValueError,TypeError,SyntaxError,ZeroDivisionError,RecursionError,OverflowError):
  return dict(correct=False,has_box=boxed,expression=ex,valid_expression=False,inventory_valid=False,value=None)
