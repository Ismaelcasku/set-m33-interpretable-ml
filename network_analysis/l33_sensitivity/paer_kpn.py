import sys, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from summarize import summarize

inp,out,shap,corrected=map(Path,sys.argv[1:])
rows=[]
z=zipfile.ZipFile(inp/'SET_M33_network_analysis_final.zip')
m=pd.read_csv(inp/'tetramers_MIC_no_exact_training_overlap.csv')
f=next(inp.glob('*.fasta')).read_text().splitlines()
seqid=dict(zip(f[1::2],[i[1:] for i in f[::2]]))
x=pd.read_csv(inp/'starpep_descriptors.csv',float_precision='round_trip').set_index('ID',verify_integrity=True)
assert set(x.index)==set(seqid.values())
checks=[]; arms=[]; contacts=[]; modules=[]; topfeatures=[]
for sp,key,col in [('paeruginosa','aeruginosa','P_aeruginosa'),('kpneumoniae','kleb','K_pneumoniae')]:
 for q in (4,5,6):
  group=f'{sp}_{q}plus'
  def member(test):
   f,=[k for k in z.namelist() if f'/selected_networks/{group}/' in k and test(k.rsplit('/',1)[-1])]
   return f,pd.read_csv(z.open(f))
  fn,d=member(lambda f:f.startswith('nodes_th'))
  _,e=member(lambda f:f.startswith('edges_th'))
  _,meta=member(lambda f:f.startswith(sp) and f.endswith('.csv') and 'CANDIDATES' not in f)
  d=d.set_index('Id',verify_integrity=True); meta=meta.set_index('Id',verify_integrity=True).loc[d.index]
  sq=d.Sequence
  threshold=float(fn.split('nodes_th')[1][:-4])
  h=pd.read_csv(next(inp.glob(f'filtered_{key}{q}*.csv'))).set_index('ID',verify_integrity=True)
  sc=StandardScaler().fit(h)
  S=cosine_similarity(sc.transform(h)); i=h.index.get_indexer(e.Source); j=h.index.get_indexer(e.Target)
  assert min(i)>=0 and min(j)>=0
  adj=np.zeros(S.shape,bool); adj[i,j]=adj[j,i]=True
  diff=int(np.triu((S>=threshold)!=adj,1).sum()); err=float(abs(S[i,j]-e.Weight).max())
  assert diff==0 and err<=1e-10 and set(h.index[adj.any(axis=1)])==set(d.index),group
  del S,adj
  ids=sq[sq.isin(seqid)].index
  a=x.loc[sq.loc[ids].map(seqid),h.columns]; b=h.loc[ids]
  slopes=[]; offsets=[]; residuals=[]
  for c in h:
   u,v=a[c].to_numpy(),b[c].to_numpy()
   assert np.isfinite(u).all() and np.isfinite(v).all() and np.ptp(u)>0,(group,c)
   s,o=np.polyfit(u,v,1); r=float(max(abs(s*u+o-v)))
   assert s>0 and r<=1e-8,(group,c,s,r)
   slopes.append(s);offsets.append(o);residuals.append(r)
  checks.append(dict(group=group,threshold=threshold,n_edges=len(e),edge_disagreements=diff,max_weight_error=err,n_anchors=len(ids),max_descriptor_error=max(residuals)))
  ref=sc.transform(h.loc[d.index])
  anchor=sc.transform(x.loc[[seqid['KKIRVRLSA']],h.columns]*slopes+offsets)[0]
  deviation=pd.DataFrame(np.abs(ref-anchor),index=d.index,columns=h.columns)
  variants={'full': list(h.columns)}
  sources={'top10_historical':shap/group/'reference_importance.csv'}
  if group == 'kpneumoniae_5plus':
   sources['top10_updated']=corrected
  for name,path in sources.items():
   rank=pd.read_csv(path).sort_values(['mean_abs_shap_full','descriptor'],ascending=[False,True]).head(10)
   variants[name]=[c for c in rank.descriptor if c in h]
  rows.extend(summarize(group,meta.loc[sq.ne('KKIRVRLSA')],deviation,variants))
pd.DataFrame(rows).to_csv(out,index=False)
