import sys,zipfile
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

inp,out,shap=map(Path,sys.argv[1:])
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
  cohort=m[m[col].notna()&m.sequence.str.count('[KRH]').eq(q)&~m.sequence.isin(sq)]
  query=sc.transform(x.loc[cohort.sequence.map(seqid),h.columns]*slopes+offsets)
  assert np.isfinite(query).all()
  sims=cosine_similarity(query,ref)
  for row,sim in zip(cohort.itertuples(),sims):
   best=int(sim.argmax()); hit=np.flatnonzero(sim>=threshold)
   arms.append(dict(group=group,peptide_id=row.peptide_id,sequence=row.sequence,MIC_tetramer_uM=getattr(row,col),n_contacts=len(hit),max_cosine=sim[best],nearest_node=d.index[best]))
   for k in hit:
    node=meta.iloc[k]
    contacts.append(dict(group=group,peptide_id=row.peptide_id,neighbor_id=d.index[k],cosine=sim[k],community=node.modularity_class,pagerank=node.pageranks,archived_neighbor_mic_pred=node.mic_pred))
  rank=pd.read_csv(shap/group/'reference_importance.csv').sort_values(['mean_abs_shap_full','descriptor'],ascending=[False,True]).head(10)
  assert len(rank)==10 and rank.descriptor.is_unique and np.isfinite(rank.mean_abs_shap_full).all(),group
  features=[c for c in rank.descriptor if c in h]
  for k,c in enumerate(rank.descriptor,1):
   topfeatures.append(dict(group=group,rank=k,descriptor=c,in_network=c in h))
  if not features: continue
  anchor=sc.transform(x.loc[[seqid['KKIRVRLSA']],h.columns]*slopes+offsets)[0]
  assert np.isfinite(anchor).all()
  ix=h.columns.get_indexer(features)
  dist=pd.Series(np.abs(ref[:,ix]-anchor[ix]).mean(axis=1),index=d.index)
  sub=meta.assign(distance=dist).loc[sq.ne('KKIRVRLSA')]
  batch=[]
  for mod,g in sub.groupby('modularity_class'):
   high=g.pageranks.eq(g.pageranks.max())
   if not (~high).any():continue
   batch.append(dict(group=group,module=mod,n_max_pagerank=int(high.sum()),n_other=int((~high).sum()),n_descriptors=len(features),mean_predicted_mic=g.mic_pred.mean(),delta=g.loc[high,'distance'].mean()-g.loc[~high,'distance'].mean()))
  batch.sort(key=lambda r:(r['mean_predicted_mic'],r['module']))
  k=len(batch)//2
  for j,r in enumerate(batch):
   r['mic_group']='lower' if j<k else ('higher' if j>=len(batch)-k else 'middle')
  modules.extend(batch)
for name,rows in [('network_validation',checks),('arm_projection',arms),('arm_contacts',contacts),('l33_modules',modules),('top_shap_overlap',topfeatures)]:
 pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
print(pd.DataFrame(arms).groupby('group').n_contacts.agg(n='size',covered=lambda a:(a>0).sum()))
if modules:
 print(pd.DataFrame(modules).groupby(['group','mic_group']).delta.agg(n='size',mean='mean',closer=lambda a:(a<0).sum()))
