"""Explicit replacement analysis for S2.1-S2.3; never overwrite prior outputs."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SPECIES = {'ecoli': {4:250,5:1874,6:1667,7:127},
           'paeruginosa': {4:3620,5:1199,6:526,7:38},
           'kpneumoniae': {4:798,5:211,6:165}}
N_REPEATS = 20
SEED = 42

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def effective_dimension(x):
    # Covariance eigenvalues are proportional to squared centered singular values.
    eigen = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)**2
    denom = np.square(eigen).sum()
    return float(eigen.sum()**2 / denom) if denom > 0 else np.nan

def cluster_metrics(points, labels):
    """Euclidean, point-weighted centroid radius; silhouette excludes noise."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    assigned = labels >= 0
    clusters = np.unique(labels[assigned])
    distances = []
    for label in clusters:
        block = points[labels == label]
        distances.extend(np.linalg.norm(block - block.mean(axis=0), axis=1))
    n_assigned = int(assigned.sum())
    valid = 2 <= len(clusters) < n_assigned
    reason = ('valid' if valid else 'fewer_than_two_clusters' if len(clusters)<2
              else 'one_cluster_per_assigned_point')
    return dict(k=len(clusters), noise=float((~assigned).mean()),
                n_assigned=n_assigned, assigned_fraction=float(assigned.mean()),
                majority_noise=bool((~assigned).mean() > 0.5),
                intra_r=float(np.mean(distances)) if distances else np.nan,
                sil=float(silhouette_score(points[assigned],labels[assigned],metric='euclidean')) if valid else np.nan,
                silhouette_valid=valid, silhouette_status=reason)

def summarize(runs, species, charge, n, nsub, dim):
    result = dict(species=species, charge=charge, n=n, n_sub=nsub,
                  effective_dimension=dim, repeats=N_REPEATS)
    for space in ['UMAP','PCA']:
        part=runs.loc[(runs.charge==charge)&(runs.space==space)]
        for metric in ['k','noise','intra_r','sil','assigned_fraction']:
            values=part[metric].dropna()
            result[f'{space}_{metric}_mean']=float(values.mean()) if len(values) else np.nan
            result[f'{space}_{metric}_sd']=float(values.std(ddof=1)) if len(values)>1 else np.nan
            result[f'{space}_{metric}_n_valid']=len(values)
        result[f'{space}_majority_noise_runs']=int(part.majority_noise.sum())
        result[f'{space}_zero_cluster_runs']=int((part.k==0).sum())
        result[f'{space}_one_cluster_runs']=int((part.k==1).sum())
        result[f'{space}_assigned_fraction_min']=float(part.assigned_fraction.min())
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--species',choices=list(SPECIES),required=True)
    parser.add_argument('--allow-unpinned',action='store_true',help='Local validation only; recorded in manifest.')
    args=parser.parse_args()
    bundle=args.bundle.resolve(); src=bundle/'inputs'; out=args.out.resolve(); species=args.species
    require(not out.exists(),'Output already exists; choose a new directory.')
    require(not out.is_relative_to(src),'Output must be outside inputs.')
    expected={'numpy':'1.26.4','pandas':'2.2.3','scikit-learn':'1.5.2','hdbscan':'0.8.40'}
    versions={name:importlib.metadata.version(name) for name in expected}
    require(args.allow_unpinned or versions==expected,f'Activate the pinned project environment: {versions}')
    import hdbscan
    manifest=json.loads((bundle/'input_hashes.json').read_text())
    for name,digest in manifest.items():
        require(sha(src/name)==digest,f'Input hash mismatch: {name}')
    base=src/'historical_umap'/species
    x=np.load(base/'grouped_shap.npy').astype(np.float64)
    rows=pd.read_csv(base/'row_order.csv',dtype={'ID':str})
    primary=json.loads((base/'run.json').read_text())
    require(primary.get('status')=='completed','Primary UMAP did not complete.')
    umap=np.load(base/'historical_embedding.npy').astype(np.float64)
    embedding_csv=pd.read_csv(base/'historical_embedding.csv',dtype={'ID':str})
    counts=SPECIES[species]; charges=rows.cationic_charge.to_numpy(int)
    require(rows.groupby('cationic_charge').size().to_dict()==counts,'Unexpected candidate counts.')
    require(x.shape==(len(rows),3 if species=='kpneumoniae' else 6),'Unexpected grouped SHAP shape.')
    require(umap.shape==(len(rows),2),'Unexpected UMAP shape.')
    require(np.isfinite(x).all() and np.isfinite(umap).all(),'Nonfinite coordinates.')
    require(rows.ID.is_unique and rows.ID.tolist()==embedding_csv.ID.tolist(),'Invalid UMAP row alignment.')
    require(np.array_equal(charges,embedding_csv.cationic_charge.to_numpy(int)),'UMAP charge mismatch.')
    require(np.allclose(umap,embedding_csv[['UMAP1','UMAP2']],rtol=1e-6,atol=1e-6),'UMAP coordinate mismatch.')
    dimensions={}
    for charge,n in counts.items():
        raw=src/'shap_reference'/f'{species}_{charge}plus'
        full_shap=np.load(raw/'shap_values.npy').astype(np.float64)
        features=json.loads((raw/'features.json').read_text())
        raw_rows=pd.read_csv(raw/'rows.csv',dtype={'ID':str})
        require(full_shap.shape==(n,len(features)),f'Unexpected full SHAP shape for {species} +{charge}.')
        require(np.isfinite(full_shap).all(),'Nonfinite full SHAP values.')
        require(raw_rows.ID.tolist()==rows.loc[charges==charge,'ID'].tolist(),'Raw SHAP row mismatch.')
        dimensions[charge]=effective_dimension(full_shap)
    # Fit once jointly to ALL candidates, without scaling/whitening, matching
    # the grouped SHAP space used by the current primary UMAP figures.
    pca_model=PCA(n_components=2,svd_solver='full',whiten=False)
    pca=pca_model.fit_transform(x)
    nsub=min(counts.values()); metrics=[]; sampling=[]; labels_out=[]
    species_index=list(SPECIES).index(species)
    for repeat in range(N_REPEATS):
        for charge in counts:
            pool=np.flatnonzero(charges==charge)
            rng=np.random.default_rng(np.random.SeedSequence([SEED,species_index,repeat,charge]))
            idx=np.sort(rng.choice(pool,size=nsub,replace=False))
            sampling.extend(dict(repeat=repeat,charge=charge,row_index=int(i),ID=rows.ID.iloc[i]) for i in idx)
            for space,coordinates in [('UMAP',umap),('PCA',pca)]:
                points=coordinates[idx]
                labels=hdbscan.HDBSCAN(min_cluster_size=5,min_samples=5,metric='euclidean',
                                      cluster_selection_method='eom',allow_single_cluster=False,
                                      approx_min_span_tree=False,core_dist_n_jobs=1).fit_predict(points)
                m=cluster_metrics(points,labels)
                metrics.append(dict(species=species,repeat=repeat,charge=charge,space=space,n_sub=nsub,**m))
                labels_out.extend(dict(repeat=repeat,charge=charge,space=space,row_index=int(i),cluster=int(label)) for i,label in zip(idx,labels))
    runs=pd.DataFrame(metrics)
    table=pd.DataFrame([summarize(runs,species,c,n,nsub,dimensions[c]) for c,n in counts.items()])
    require(len(runs)==N_REPEATS*len(counts)*2,'Incomplete calculation.')
    require(all(sha(src/name)==digest for name,digest in manifest.items()),'Inputs modified.')
    out.mkdir(parents=True)
    table.to_csv(out/'table_full_precision.csv',index=False,float_format='%.17g')
    runs.to_csv(out/'metrics_by_draw.csv',index=False,float_format='%.17g')
    pd.DataFrame(sampling).to_csv(out/'sampled_rows.csv',index=False)
    pd.DataFrame(labels_out).to_csv(out/'cluster_labels.csv.gz',index=False)
    rows.assign(PCA1=pca[:,0],PCA2=pca[:,1]).to_csv(out/'PCA_coordinates.csv',index=False,float_format='%.17g')
    columns={'charge':'Charge','n':'n','n_sub':'n sub','effective_dimension':'Eff. dim'}
    for space in ['UMAP','PCA']:
        columns.update({f'{space}_{k}_mean':f'{space} {label}' for k,label in [('k','k'),('noise','noise'),('intra_r','intra-r'),('sil','sil.')]})
        columns[f'{space}_sil_n_valid']=f'{space} valid sil. / 20'
    display=table[list(columns)].rename(columns=columns).copy()
    for col in display:
        if col in ['Charge','n','n sub'] or 'valid sil.' in col:
            display[col]=display[col].map(lambda v:str(int(v)))
        else:
            display[col]=display[col].map(lambda v:'NA' if pd.isna(v) else f'{v:.1f}' if col.endswith(' k') else f'{v:.2f}')
    display.to_csv(out/'table_for_Word.tsv',sep='\t',index=False)
    md='| '+' | '.join(display.columns)+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'
    md+='\n'.join('| '+' | '.join(row)+' |' for row in display.to_numpy())+'\n'
    (out/'table.md').write_text(md)
    report=dict(status='completed',species=species,counts=counts,n_sub=nsub,repeats=N_REPEATS,
                seed=SEED,packages=versions,pinned_environment=versions==expected,python=platform.python_version(),
                input_sha256=manifest,inputs_unchanged=True,umap_refitted=False,
                pca_explained_variance_ratio=pca_model.explained_variance_ratio_.tolist(),
                hdbscan=dict(min_cluster_size=5,min_samples=5,metric='euclidean',cluster_selection_method='eom',
                             allow_single_cluster=False,approx_min_span_tree=False,core_dist_n_jobs=1),
                method='METHODS.md',not_a_reproduction_of_legacy_tables=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(md,flush=True)

if __name__=='__main__':
    main()
