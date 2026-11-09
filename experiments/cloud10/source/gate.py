"""Conservative Python return-event detector; sees generated code only, never task or gold."""
import ast,io,tokenize

def inspect_return(generated):
 text=generated.lstrip();fenced=text.startswith('```')
 if fenced:
  if '\n' not in text:return None
  header,code=text.split('\n',1)
  if header.strip() not in ('```python','```py','```'):return None
  if '```' in code:return None
 else:code=text
 code=code.rstrip(' \t\r\n')
 if not code.endswith('return'):return None
 try:
  toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
  sig=[t for t in toks if t.type not in (tokenize.NEWLINE,tokenize.NL,tokenize.INDENT,tokenize.DEDENT,tokenize.ENDMARKER)]
  if not sig or sig[-1].type!=tokenize.NAME or sig[-1].string!='return':return None
  t=sig[-1]
  # Complete snapshot must already be syntactically legal. No synthetic completion or answer token.
  tree=ast.parse(code);compile(tree,'<generated-return-gate>','exec')
  matches=[n for n in ast.walk(tree) if isinstance(n,ast.Return) and n.value is None and n.lineno==t.start[0] and ast.get_source_segment(code,n)=='return']
  if not matches:return None
  n=matches[-1]
  return dict(event='LEGAL_RETURN_READY',line=n.lineno,column=n.col_offset,code_characters=len(code),fenced=fenced,detector='tokenize+AST+compile of generated prefix; no execution',generated_prefix=generated)
 except (SyntaxError,ValueError,tokenize.TokenError,IndentationError):return None
