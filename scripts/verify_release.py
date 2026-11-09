"""Verify public file hashes, artifact shapes and local documentation links."""
from pathlib import Path
import hashlib,json,re,ast
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'manifest.json').read_text());errors=[]
 for name,v in m['files'].items():
  p=ROOT/name
  if not p.is_file():errors.append('missing '+name);continue
  if hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:errors.append('hash '+name)
  if p.stat().st_size!=v['bytes']:errors.append('size '+name)
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['.git','.venv','__pycache__','reproduction-output']) and p.name!='manifest.json'}
 expected={k for k in m['files'] if Path(k).name!='manifest.json'}
 errors.extend('unmanifested '+x for x in sorted(actual-expected))
 for p in ROOT.rglob('*.py'):
  if any(x in p.parts for x in ['.git','.venv','reproduction-output','__pycache__']):continue
  ast.parse(p.read_text(),filename=str(p))
 for p in ROOT.rglob('*.md'):
  for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
   if target.startswith(('http:','https:','mailto:','#')):continue
   target=target.split('#')[0]
   if target and not (p.parent/target).exists():errors.append('link '+str(p.relative_to(ROOT))+': '+target)
 if errors:raise SystemExit('\n'.join(errors))
 print(f"Verified {len(m['files'])} hashes, Python syntax and local Markdown links. Historical manifests are evidence, not public-export checksums.")
if __name__=='__main__':main()
