"""Replay original WEKA models, then score only the fixed experimental panel."""
import csv
import io
import math
import random
import re
import shlex
import statistics as st
import subprocess as sp
import sys
import tarfile
from pathlib import Path

MODELS = [
    ('paeruginosa','P_aeruginosa','aeruginosa',4,10,'batch4pseudo10',3620),
    ('paeruginosa','P_aeruginosa','aeruginosa',5,20,'batch5pseudo20',1199),
    ('paeruginosa','P_aeruginosa','aeruginosa',6,10,'batch6pseudo10',526),
    ('kpneumoniae','K_pneumoniae','klebsiella',4,10,'batch4kleb10',798),
    ('kpneumoniae','K_pneumoniae','klebsiella',6,60,'batch6kleb60',165),
]

def read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save(path, data):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(data[0]))
        w.writeheader()
        w.writerows(data)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def attrs(header):
    return [shlex.split(s)[1] for s in header.splitlines()
            if s.strip().lower().startswith('@attribute ')]

def arff(path, header, rows):
    with path.open('w') as f:
        f.write(header.rstrip() + '\n')
        for row in rows:
            require(all(x == '?' or math.isfinite(float(x)) for x in row), path)
            f.write(','.join(map(str,row))+'\n')

def fit(xs, ys):
    a,b = st.mean(xs),st.mean(ys)
    s = sum((x-a)**2 for x in xs)
    require(s > 0, 'Constant anchor descriptor')
    m = sum((x-a)*(y-b) for x,y in zip(xs,ys))/s
    require(m > 0, 'Nonpositive slope')
    return m, b-m*a

def run(out, source=None, jar=None, headers=None):
    seqid = {}
    for record in Path('inputs/arms.fasta').read_text().split('>')[1:]:
        name,*seq = record.splitlines()
        seqid.setdefault(''.join(seq), name.split()[0])
    raw = {r['ID']:r for r in read('inputs/descriptors.csv')}
    mic = read('inputs/mic.csv')
    ids = {(r['group'],r['ID']) for r in read('inputs/anchor_ids.csv')}
    results, checks, summary = [], [], []
    java = ['java','-cp',str(out/'classes')+':'+str(jar),'FrozenRF']
    if not headers:
        (out/'classes').mkdir()
        sp.run(['javac','-cp',str(jar),'-d',str(out/'classes'),'FrozenRF.java'],check=True)
    with tarfile.open('inputs/paer_kpn_weka_inputs.tar.gz') as archive:
        for species,col,stem,q,pool,model,n in MODELS:
            ref = 'SET{}_filtrado_{}_pool{}_con_prediccion.csv'.format(stem,q,pool)
            rows = list(csv.DictReader(io.TextIOWrapper(
                archive.extractfile(species+'/'+ref),encoding='utf-8-sig')))
            require(len(rows)==n, model+' unexpected replay count')
            if headers:
                header = (headers/(model+'.arff')).read_text()
            else:
                suffix = 'Training/WEKA_models/{}/batch {}/{}.model'.format(species,q,model)
                matches = [p for p in source.rglob(model+'.model') if str(p).endswith(suffix)]
                require(len(matches)==1, 'Cannot uniquely locate '+suffix)
                path = matches[0]
                header = sp.check_output(java+['header',str(path)],universal_newlines=True)
            names = attrs(header)
            require(names.count('MIC')==1 and 'ID' not in names and len(set(names))==len(names), model)
            features = [k for k in names if k!='MIC']
            anchors = [r for r in rows if r['ID'] in raw and (species+'_'+str(q)+'plus',r['ID']) in ids]
            random.Random(42).shuffle(anchors)
            # The archived +4 extreme defines the scale for a nearly constant descriptor.
            if model=='batch4pseudo10':
                anchors.sort(key=lambda r: r['ID']!='StarPep_starPep_01566_008_3')
            split = int(.75*len(anchors))
            require(split>=2 and len(anchors)-split>=2, 'Too few anchors '+model)
            coeff = []
            for key in features:
                x = [float(raw[r['ID']][key]) for r in anchors]
                y = [float(r[key]) for r in anchors]
                slope,intercept = fit(x[:split],y[:split])
                err = max(abs(slope*a+intercept-b) for a,b in zip(x[split:],y[split:]))
                require(err<=1e-8,(model,key,err))
                coeff.append(dict(descriptor=key,slope=slope,intercept=intercept,holdout_error=err))
            save(out/(model+'_scaling.csv'),coeff)
            replay = out/(model+'_replay.arff')
            arff(replay,header,(['?' if k=='MIC' else r[k] for k in names] for r in rows))
            cohort = [r for r in mic if r[col].strip() and sum(r['sequence'].count(c) for c in 'KRH')==q]
            require(all(float(r[col])>0 for r in cohort), model+' invalid MIC')
            require(len({r['peptide_id'] for r in cohort})==len(cohort), model+' duplicate peptide ID')
            projected = []
            for r in cohort:
                desc = raw[seqid[r['sequence']]]
                values = {c['descriptor']:c['slope']*float(desc[c['descriptor']])+c['intercept'] for c in coeff}
                projected.append(['?' if k=='MIC' else values[k] for k in names])
            query = out/(model+'_tetramers.arff')
            arff(query,header,projected)
            save(out/(model+'_row_map.csv'), [dict(inst=i,peptide_id=r['peptide_id']) for i,r in enumerate(cohort,1)])
            print(model,len(anchors),'anchors;',len(cohort),'tetramers',flush=True)
            if headers:
                continue
            def predict(test, count):
                s = sp.check_output(java+['predict',str(path),str(test)],universal_newlines=True)
                pred = list(csv.DictReader(io.StringIO(s)))
                require([int(r['inst#']) for r in pred]==list(range(1,count+1)), model+' prediction IDs')
                return [float(r['predicted']) for r in pred]
            rp = predict(replay,len(rows))
            err = max(abs(v-float(r['predicted'])) for v,r in zip(rp,rows))
            require(err<=0.00050001,(model,'historical replay failed',err))
            pp = predict(query,len(cohort))
            checks.append(dict(model=model,n_replay=len(rows),max_replay_error=err,
                               n_fit=split,n_holdout=len(anchors)-split,
                               max_holdout_error=max(c['holdout_error'] for c in coeff)))
            obs = [float(r[col]) for r in cohort]
            for r,y,v in zip(cohort,obs,pp):
                results.append(dict(species=col,charge=q,model=model,peptide_id=r['peptide_id'],
                                    MIC_tetramer_uM=y,rf_output=v))
            # Within-model rank comparison is independent of output units.
            pairs = [((obs[i]-obs[j])*(pp[i]-pp[j])) for i in range(len(obs))
                     for j in range(i) if max(obs[i],obs[j])/min(obs[i],obs[j])>2]
            summary.append(dict(species=col,charge=q,n=len(obs),pairs_MIC_ratio_gt_2=len(pairs),
                                concordant=sum(x>0 for x in pairs),discordant=sum(x<0 for x in pairs),
                                ties_pred=sum(x==0 for x in pairs)))
    if not headers:
        save(out/'validation.csv',checks)
        save(out/'predictions.csv',results)
        save(out/'ranking_by_charge.csv',summary)

if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    if sys.argv[2]=='--prepare-only':
        run(out,headers=Path(sys.argv[3]))
    else:
        run(out,Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
