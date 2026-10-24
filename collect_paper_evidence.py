#!/usr/bin/env python3
"""Conservative, auditable collection of manuscript-related experimental evidence.
Run on the Windows computer where E: is accessible. Python standard library only.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, re, shutil
from collections import Counter
from pathlib import Path

SOURCE = Path(r'E:\47\472\New_Papers\A Data-Centric Structural Biomedical Signal Processing and Control_Submitted\Final\Evidence')
DEST = Path(r'E:\47\472\New_Papers\A Data-Centric Structural Biomedical Signal Processing and Control_Submitted\Final\A Data-Centric Machine Learning Framework_Diagnostics\Modified 2\New_Experiments')
GROUPS = {
 '01_Data_and_Leakage': ['nhanes','hrs','pfq061','disability_stage','disability_binary','leakage','imput','preprocess','harmoniz','crosswalk','target_defin','participant'],
 '02_Graph_and_Views': ['relation_graph','dependency_graph','constraint','forbidden_pair','jaccard','view_generation','view_partition','graph_edge','view_composition'],
 '03_Rank_and_Conditioning': ['singular_value','condition_number','conditioning','matrix_rank','spectral','svd','rank_analysis'],
 '04_Fusion_and_Stacking': ['out_of_fold','oof','stacking','weighted_fusion','multiview','multi_view','catboost','fusion','meta_learner'],
 '05_Diffusion': ['diffusion','structural_energy','regularization','smoothing'],
 '06_Validation_and_Ablations': ['temporal','ablation','roc_auc','macro_f1','balanced_accuracy','pr_auc','boosting','nested_cv','validation'],
 '07_NHANES_to_HRS': ['nhanes_to_hrs','nhanes2hrs','external_transfer','cross_dataset','domain_shift','hrs_disability_binary_candidate'],
 '08_HRS_Native': ['hrs_native','hrs_only','hrs_xgboost','hrs_benchmark'],
}
CODE = {'.py','.ipynb','.r','.rmd','.m','.jl','.sh','.bat','.ps1','.sql'}
DATA = {'.csv','.tsv','.parquet','.feather','.xlsx','.xls','.jsonl','.sav','.dta','.sas7bdat','.xpt','.h5','.hdf5','.npz','.npy','.pkl','.pickle','.joblib'}
RESULTS = {'.json','.yaml','.yml','.txt','.md','.log','.png','.jpg','.jpeg','.svg','.pdf','.html','.tex','.xml','.toml'}
SKIP_DIRS = {'__pycache__','.git','.venv','venv','node_modules','.ipynb_checkpoints','site-packages'}
IGNORE_EXT = {'.exe','.dll','.pyc','.tmp','.bak','.zip','.7z','.rar'}
# Exact manuscript markers receive higher weight than broad keywords.
MARKERS = ['disability_binary_merge01_vs_23','hrs_disability_binary_candidate','pfq061k','pfq061l','pfq061b','pfq061c','pfq061m','pfq061p','pfq061e','upper_limb_difficulty','mobility_difficulty','adl_difficulty','phq9_score','se qn_cycle_key'.replace(' ',''),'diffusion','nhanes','hrs']

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def text_sample(path, max_bytes):
 if path.suffix.lower() not in CODE|{'.json','.yaml','.yml','.md','.txt','.toml','.log','.csv','.tsv','.rmd'}:return ''
 try:
  with path.open('rb') as f: raw=f.read(max_bytes)
  return raw.decode('utf-8',errors='replace').lower()
 except (OSError,UnicodeError):return ''

def classify(path, relative, max_bytes):
 ext=path.suffix.lower()
 if ext in CODE:kind='Code'
 elif ext in DATA:kind='Data'
 elif ext in RESULTS:kind='Results_or_Config'
 else:return None
 name=str(relative).lower().replace(' ','_').replace('-','_')
 body=text_sample(path,max_bytes)
 scores={}
 for group,terms in GROUPS.items():
  scores[group]=sum((5 if t in name else 0)+(2 if t in body else 0) for t in terms)
 marker_hits=[m for m in MARKERS if m in name or m in body]
 score=max(scores.values(),default=0)+(min(len(marker_hits),5)*2)
 # Generic result files are accepted only when path or contents link to this study.
 if score<5:return None
 group=max(scores,key=scores.get)
 confidence='high' if score>=13 else 'medium' if score>=8 else 'review'
 return dict(kind=kind,group=group,score=score,confidence=confidence,markers=marker_hits)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source',type=Path,default=SOURCE)
 ap.add_argument('--dest',type=Path,default=DEST)
 ap.add_argument('--copy',action='store_true',help='Actually copy; default is dry-run')
 ap.add_argument('--include-review',action='store_true',help='Also copy low-confidence candidates')
 ap.add_argument('--max-scan-mb',type=int,default=2)
 ap.add_argument('--max-file-gb',type=float,default=10)
 args=ap.parse_args()
 src=args.source.resolve(); dst=args.dest.resolve()
 if not src.is_dir():ap.error(f'Source folder does not exist: {src}')
 if dst==src or src in dst.parents:ap.error('Destination cannot be inside source')
 records=[]; ignored=Counter(); errors=[]
 for root,dirs,files in __import__('os').walk(src):
  dirs[:]=sorted(d for d in dirs if d.lower() not in SKIP_DIRS)
  for filename in sorted(files):
   p=Path(root)/filename; rel=p.relative_to(src)
   try:
    if p.is_symlink() or p.suffix.lower() in IGNORE_EXT:ignored['symlink_or_ignored_type']+=1;continue
    size=p.stat().st_size
    if size>args.max_file_gb*1024**3:ignored['over_size_limit']+=1;continue
    match=classify(p,rel,args.max_scan_mb*1024**2)
    if not match:ignored['not_matched']+=1;continue
    action='selected' if match['confidence']!='review' or args.include_review else 'review_only'
    records.append({'relative_path':str(rel),'size_bytes':size,'action':action,**match})
   except (OSError,ValueError) as e:errors.append({'path':str(rel),'error':str(e)})
 # A file is never silently overwritten; existing identical copies are verified.
 dst.mkdir(parents=True,exist_ok=True)
 for row in records:
  if row['action']!='selected':continue
  origin=src/row['relative_path']; target=dst/'Collected'/row['relative_path']
  if not args.copy:row['status']='would_copy';continue
  try:
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    if digest(origin)==digest(target):row['status']='already_identical'
    else:row['status']='conflict_not_overwritten'
   else:
    shutil.copy2(origin,target)
    row['status']='copied_verified' if digest(origin)==digest(target) else 'COPY_VERIFICATION_FAILED'
   row['sha256_source']=digest(origin)
  except OSError as e:row['status']='copy_error';row['error']=str(e)
 manifest=dst/'collection_manifest.csv'
 fields=['relative_path','kind','group','score','confidence','markers','size_bytes','action','status','sha256_source','error']
 with manifest.open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in records:
   writer.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),list) else row.get(k,'') for k in fields})
 summary={'source':str(src),'destination':str(dst),'mode':'copy' if args.copy else 'dry_run',
          'candidates':len(records),'selected':sum(r['action']=='selected' for r in records),
          'by_group':dict(Counter(r['group'] for r in records if r['action']=='selected')),
          'by_kind':dict(Counter(r['kind'] for r in records if r['action']=='selected')),
          'ignored':dict(ignored),'errors':errors,
          'limitations':['Heuristic selection, not proof of reproduction.',
                         'Imported dependencies and dynamically generated paths require manual inspection.',
                         'No experiment is executed and no reported metric is verified.',
                         'Review collection_manifest.csv for omissions and false positives.']}
 (dst/'collection_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps(summary,indent=2,ensure_ascii=False))
 print('Manifest:',manifest)

if __name__=='__main__':main()
