"""Stage complete-library inputs and unchanged references into a new workspace."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from setm33robust.config import load_tasks

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate-root',type=Path,required=True,
               help='Candidate_libraries directory extracted from package 02')
p.add_argument('--archive-root',type=Path,required=True,
               help='Post_WEKA_robustness directory extracted from package 05')
p.add_argument('--work',type=Path,required=True)
a = p.parse_args()
work = a.work.resolve()
if work.exists():
    raise FileExistsError('Choose a new workspace; existing results are never modified')
tasks = load_tasks()
sources = []
manifest = {name.split('Test/Analogs predictions/',1)[1]:sha
            for sha,name in (line.split('  ',1) for line in (ROOT/'data/input_manifest.sha256').read_text().splitlines())}
for _, task in tasks.iterrows():
    rel = task.prediction.split('Test/Analogs predictions/',1)[1]
    source = a.candidate_root/'Archived_WEKA_predictions'/rel
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[rel]:
        raise ValueError(f'Input checksum mismatch: {task.key}')
    if task.key == 'ecoli_5plus':
        frame = pd.read_csv(source)
        if len(frame)!=1874 or not frame.ID.is_unique:
            raise ValueError('Expected 1,874 distinct E. coli +5 candidates')
    sources.append((source,Path('data/Supplementary Data')/task.prediction))
for _,task in tasks.iterrows():
    if task.key != 'ecoli_5plus':
        source = a.archive_root/'shap_reference'/task.key
        if not source.is_dir(): raise FileNotFoundError(source)
        sources.append((source,Path('results/shap_reference')/task.key))
for species in ['paeruginosa','kpneumoniae']:
    parent = 'historical_umap' if (a.archive_root/'historical_umap').is_dir() else 'primary_umap'
    source = a.archive_root/parent/species
    if not source.is_dir(): raise FileNotFoundError(source)
    sources.append((source,Path('results/historical_umap')/species))
work.mkdir(parents=True)
for source, rel in sources:
    dst = work/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir(): shutil.copytree(source,dst)
    else: shutil.copy2(source,dst)
for species in ['paeruginosa','kpneumoniae']:
    base = work/'results/historical_umap'/species
    for ext in ['csv','npy']:
        old = base/f'primary_embedding.{ext}'
        if old.exists(): old.rename(base/f'historical_embedding.{ext}')
print(f'Prepared new workspace: {work}')
print('Set SET_M33_DATA_ROOT to its data/Supplementary Data directory and SET_M33_RESULTS_ROOT to its results directory.')
