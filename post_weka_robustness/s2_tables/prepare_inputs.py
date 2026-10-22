"""Prepare S2 inputs from deposited package 05, without copying results into Git."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive-root', type=Path, required=True)
p.add_argument('--work', type=Path, required=True)
a = p.parse_args()
src, work = a.archive_root.resolve(), a.work.resolve()
if work.exists():
    raise FileExistsError('Choose a new work directory')
required = []
for species, charges in [('ecoli',[4,5,6,7]), ('paeruginosa',[4,5,6,7]), ('kpneumoniae',[4,5,6])]:
    for charge in charges:
        for name in ['shap_values.npy','features.json','rows.csv']:
            rel = Path('shap_reference')/f'{species}_{charge}plus'/name
            required.append((src/rel, rel))
    base = src/('historical_umap' if (src/'historical_umap').is_dir() else 'primary_umap')/species
    for name in ['grouped_shap.npy','row_order.csv','run.json','historical_embedding.npy','historical_embedding.csv']:
        source = base/name
        if not source.exists() and name.startswith('historical_embedding'):
            source = base/name.replace('historical_embedding','primary_embedding')
        required.append((source,Path('historical_umap')/species/name))
for source, _ in required:
    if not source.is_file():
        raise FileNotFoundError(source)
work.mkdir(parents=True)
manifest = {}
for source, rel in required:
    dst = work/'inputs'/rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,dst)
    manifest[rel.as_posix()] = hashlib.sha256(dst.read_bytes()).hexdigest()
(work/'input_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copy2(Path(__file__).with_name('METHODS.md'),work/'METHODS.md')
print(f'Prepared {len(required)} input files in {work}')
