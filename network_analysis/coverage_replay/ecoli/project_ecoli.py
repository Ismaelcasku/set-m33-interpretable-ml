"""Run in a fresh directory: python project_ecoli.py INPUT_DIR OUTPUT_DIR."""
import sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

inp,out=map(Path,sys.argv[1:])
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
    cohort=mic[mic.E_coli.notna()&mic.sequence.str.count('[KRH]').eq(q)]
    assert len(cohort)=={4:12,5:13,6:3}[q]
    for pid,r in cohort.iterrows():
        same=d.sequence.eq(r.sequence).to_numpy()
        assert same.sum()<=1
        if same.any():
            idx=np.flatnonzero(same)[0]
            sim=cosine_similarity(ref[[idx]],ref)[0]
            hit=adj[idx].copy()
        else:
            v=x.loc[[pid]]*slopes+offsets
            assert np.isfinite(v.to_numpy()).all(),pid
            sim=cosine_similarity(sc.transform(v),ref)[0]
            hit=sim>=t
        sim[same]=-np.inf
        hit[same]=False
        best=int(sim.argmax())
        arms.append(dict(charge=q,peptide_id=pid,sequence=r.sequence,MIC_tetramer_uM=r.E_coli,historical_node=bool(same.any()),historical_community=d.loc[d.index[same][0],'modularity_class'] if same.any() else '',n_contacts=int(hit.sum()),nearest_node=d.index[best],max_cosine=sim[best]))
        for k in np.flatnonzero(hit):
            node=d.iloc[k]
            contacts.append(dict(charge=q,peptide_id=pid,neighbor_id=d.index[k],cosine=sim[k],community=node.modularity_class,pagerank=node.pageranks,archived_neighbor_mic_pred=node.mic_pred,archived_preselection=d.index[k] in set(cand.Id)))
out.mkdir(parents=True,exist_ok=False)
for name,rows in [('network_validation',checks),('arm_projection',arms),('arm_contacts',contacts)]:
    pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
print(pd.DataFrame(arms).groupby('charge').n_contacts.agg(['size',lambda v:(v>0).sum()]))
