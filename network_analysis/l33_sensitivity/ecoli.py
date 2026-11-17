import sys, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from summarize import summarize

inp, out, shap_zip = map(Path, sys.argv[1:])
rows=[]
def read(name,key='ID',**kw):
    return pd.read_csv(inp/name,**kw).set_index(key,verify_integrity=True)
z=zipfile.ZipFile(inp/'SET_M33_network_analysis_final.zip')
source='ecoli_projection_inputs.csv'
mic=pd.read_csv(inp/source,usecols=['peptide_id','sequence','E_coli']).dropna(subset=['peptide_id']).set_index('peptide_id',verify_integrity=True)
arms=[]; contacts=[]; checks=[]
for q,t,stem in [(4,.55,'ecoli_training_pool_20'),(5,.70,'ecoli5_training_pool_0'),(6,.70,'ecoli6_training_pool_0')]:
    h=read('filtered_'+stem+'.csv')
    x=read(source,usecols=['ID',*h.columns],float_precision='round_trip').reindex(columns=h.columns)
    f,=[k for k in z.namelist() if f'/ecoli_{q}plus/Ecoli' in k and k.endswith('.csv') and 'CANDIDATES' not in k]
    d=pd.read_csv(z.open(f)).set_index('Id',verify_integrity=True)
    cand=pd.read_csv(z.open(f[:-4]+'_CANDIDATES.csv'))
    e=pd.read_csv(z.open(f.rsplit('/',1)[0]+f'/edges_th{t:.2f}.csv'))
    ids=h.index.intersection(x.index)
    assert len(ids)=={4:250,5:715,6:1667}[q]
    a=x.loc[ids]; b=h.loc[ids]; slopes=[]; offsets=[]; errors=[]
    for c in h:
        u,v=a[c].to_numpy(),b[c].to_numpy()
        ok=np.isfinite(u)&np.isfinite(v)
        assert ok.sum()>=2 and np.ptp(u[ok])>0,c
        s,o=np.polyfit(u[ok],v[ok],1)
        err=max(abs(s*u[ok]+o-v[ok]))
        assert s>0 and err<=1e-8,(q,c,s,err)
        slopes.append(s); offsets.append(o); errors.append(err)
    sc=StandardScaler().fit(h); ref=sc.transform(h.loc[d.index])
    rebuilt=sc.transform(x.loc[d.index]*slopes+offsets)
    S=cosine_similarity(rebuilt)
    i,j=d.index.get_indexer(e.Source),d.index.get_indexer(e.Target)
    assert min(i)>=0 and min(j)>=0
    adj=np.zeros(S.shape,bool); adj[i,j]=adj[j,i]=True
    diff=np.triu((S>=t)!=adj,1).sum()
    delta=max(abs(S[i,j]-e.Weight))
    assert diff==0 and delta<=1e-10,(q,diff,delta)
    checks.append(dict(charge=q,threshold=t,n_edges=len(e),edge_disagreements=int(diff),max_weight_error=delta,max_descriptor_error=max(errors)))
    parent='StarPep_starPep_02423_014_4'
    anchor=sc.transform(x.loc[[parent]]*slopes+offsets)[0]
    dev=pd.DataFrame(np.abs(ref-anchor),index=d.index,columns=h.columns)
    dd=d.drop(index=parent,errors='ignore')
    with zipfile.ZipFile(shap_zip) as sh:
        historical = pd.read_csv(sh.open(f'ecoli_{q}plus_historical.csv'))
        rank = historical.sort_values(['mean_abs_shap_full','descriptor'], ascending=[False,True]).head(10)
        variants = {'full': list(h.columns), 'top10_historical': [c for c in rank.descriptor if c in h]}
        if q == 5:
            full = pd.read_csv(sh.open('ecoli_5plus_full_1874.csv'))
            rank = full.sort_values(['mean_abs_shap_full','descriptor'], ascending=[False,True]).head(10)
            variants['top10_updated'] = [c for c in rank.descriptor if c in h]
    rows.extend(summarize(f'ecoli_{q}plus', dd, dev, variants))
pd.DataFrame(rows).to_csv(out, index=False)
