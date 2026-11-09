"""Separate syntax coverage, execution, member values, and mutability. No repairs."""
import ast,builtins,copy,contextlib,io,json,re,resource,subprocess,sys,time
from pathlib import Path
import psutil
NAMES=('abs','all','any','bool','dict','enumerate','filter','float','frozenset','int','isinstance','len','list','map','max','min','ord','range','reversed','round','set','sorted','str','sum','tuple','zip','ValueError','TypeError','KeyError','IndexError','StopIteration','OverflowError','ZeroDivisionError','type','issubclass','pow','divmod','next','iter','chr','bin','oct','hex','repr','print','slice','complex','bytes','bytearray','hash','callable')
ATTRS={n for t in (str,list,tuple,dict,set,frozenset,int,float,bool) for n in dir(t) if not n.startswith('_')}
def failure(status,error=None,format_ok=False,code=None,tree=None):
 return dict(status=status,format_ok=format_ok,executable=None,values_correct=None,mutability_correct=None,functional=False,exact_frozenset=None,joint=False,code=code,ast=tree,trials=[],failure_reasons=[dict(kind='FORMAT' if not format_ok else 'EXECUTION_OR_SUPPORT',reason=status,detail=error)])
def extract(text):
 m=re.fullmatch(r'\s*```(?:python|py)?\s*\n(.*?)\n```\s*',text,re.S)
 return m.group(1) if m else text

def evaluate(p):
 t=p['task'];code=extract(p['text'])
 if not p['complete']:return failure('TRUNCATED',code=code)
 if len(code)>30000:return failure('CODE_SIZE',code=code)
 try:tree=ast.parse(code);dump=ast.dump(tree,include_attributes=True)
 except SyntaxError as e:return failure('PARSE_OR_FORMAT',str(e),code=code)
 if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return failure('NOT_ONE_COMPLETE_FUNCTION',code=code,tree=dump)
 fnnode=tree.body[0]
 if fnnode.name!=t['function_name']:return failure('WRONG_FUNCTION_NAME',code=code,tree=dump)
 # Multiple returns, assignments, loops, branches, docstrings, annotations and defaults are allowed.
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith,ast.AsyncFunctionDef)) or (isinstance(n,ast.FunctionDef) and n.decorator_list):return failure('UNSUPPORTED_BUILTINS_ONLY',type(n).__name__,True,code,dump)
  if isinstance(n,ast.Name) and (n.id.startswith('__') or n.id in ('open','eval','exec','compile','getattr','setattr','globals','locals','vars','input','breakpoint')):return failure('UNSUPPORTED_NAME',n.id,True,code,dump)
  if isinstance(n,ast.Attribute) and n.attr not in ATTRS:return failure('UNSUPPORTED_ATTRIBUTE',n.attr,True,code,dump)
 env={'__builtins__':{n:getattr(builtins,n) for n in NAMES}};output=io.StringIO()
 try:
  with contextlib.redirect_stdout(output):exec(compile(tree,'<unchanged-generated-function>','exec'),env)
  fn=env[t['function_name']]
 except Exception as e:return failure('DEFINITION_EXECUTION_ERROR',repr(e),True,code,dump)
 trials=[];reasons=[]
 for index,test in enumerate(t['tests']):
  args=copy.deepcopy(test['args']);before=copy.deepcopy(args);expected=test['expected'];mutation=None
  try:
   with contextlib.redirect_stdout(output):value=fn(*args)
   typ=type(value).__name__;original_repr=repr(value)
   shape=type(value) in (set,frozenset,list,tuple) and all(type(v).__name__==t['member_type'] for v in value)
   unique=shape and len(value)==len(set(value));members=shape and unique and set(value)==set(expected);unchanged=args==before
   if not members:reasons.append(dict(kind='MEMBER_COMPUTATION_OR_TYPE',case=index,actual=original_repr,expected=expected))
   if not unchanged:reasons.append(dict(kind='INPUT_MUTATION',case=index))
   if t['category']=='N':
    marker='__new_member__' if t['member_type']=='str' else 987654321
    try:
     snapshot=list(value);identity=id(value);assert marker not in value
     value.add(marker);added=marker in value and id(value)==identity and set(value)==set(snapshot)|{marker} and len(value)==len(snapshot)+1
     value.remove(marker);restored=marker not in value and id(value)==identity and set(value)==set(snapshot) and len(value)==len(snapshot)
     mutation=dict(passed=bool(added and restored),add_query_ok=bool(added),remove_restore_ok=bool(restored),marker=marker,error=None)
    except Exception as e:mutation=dict(passed=False,marker=marker,error=repr(e))
    if not mutation['passed']:reasons.append(dict(kind='MUTABILITY',case=index,detail=mutation))
   trials.append(dict(case=index,case_role=test['role'],args=before,expected=expected,output=original_repr,output_type=typ,exact_frozenset=typ=='frozenset',member_shape=bool(shape),unique=bool(unique),values_correct=bool(members),input_unchanged=unchanged,mutability=mutation,error=None))
  except Exception as e:
   trials.append(dict(case=index,case_role=test['role'],args=before,expected=expected,output_type=None,exact_frozenset=False,values_correct=False,input_unchanged=args==before,mutability=None,error=repr(e)));reasons.append(dict(kind='CALL_EXECUTION_ERROR',case=index,detail=repr(e)))
 executable=all(x['error'] is None for x in trials);values=all(x['values_correct'] and x['input_unchanged'] for x in trials);mutable=all(x['mutability'] and x['mutability']['passed'] for x in trials) if t['category']=='N' else None
 functional=bool(executable and values and (mutable if t['category']=='N' else True));exact=all(x['exact_frozenset'] for x in trials)
 return dict(status='EXECUTED',format_ok=True,executable=executable,values_correct=values,mutability_correct=mutable,functional=functional,exact_frozenset=exact,joint=functional and (exact if t['category']=='E' else True),code=code,ast=dump,structure=dict(return_count=sum(isinstance(n,ast.Return) for n in ast.walk(fnnode)),loop_count=sum(isinstance(n,(ast.For,ast.While)) for n in ast.walk(fnnode)),branch_count=sum(isinstance(n,ast.If) for n in ast.walk(fnnode))),trials=trials,failure_reasons=reasons,stdout=output.getvalue()[:4000])

def score(text,task,complete=True):
 payload=json.dumps(dict(text=text,task=task,complete=complete));start=time.monotonic();p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1'});pending=payload;stopped=False
 while True:
  try:out,err=p.communicate(input=pending,timeout=.05);break
  except subprocess.TimeoutExpired:
   pending=None
   try:rss=psutil.Process(p.pid).memory_info().rss
   except psutil.NoSuchProcess:rss=0
   if rss>512*2**20 or time.monotonic()-start>3:stopped=True;p.kill();out,err=p.communicate();break
 try:r=json.loads(out) if not stopped and p.returncode==0 else failure('EXECUTOR_RESOURCE_OR_EXIT')
 except json.JSONDecodeError:r=failure('EXECUTOR_PROTOCOL')
 r['process']=dict(exit_code=p.returncode,stdout=out,stderr=err);return r
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CPU,(2,2))
 try:r=evaluate(json.load(sys.stdin))
 except BaseException as e:r=failure('EXECUTOR_FAILURE',repr(e))
 print(json.dumps(r,ensure_ascii=False))
