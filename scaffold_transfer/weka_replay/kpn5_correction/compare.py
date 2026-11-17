import csv,math,re,subprocess as sp,sys
from pathlib import Path
from run import read,save,require,attrs,arff

def main(out, jar=None):
    train=read('inputs/kleb4_training_pool_60.csv')
    names=[k for k in train[0] if k!='ID']
    require(len(train)==527 and len(names)==15 and names[-1]=='MIC' and
            sum(not r['MIC'].strip() for r in train)==2,'Training archive mismatch')
    header='@relation historical_kleb4_pool60\n'+'\n'.join('@attribute '+k+' numeric' for k in names)+'\n@data\n'
    def build(src,dst):
        h,d=re.split(r'^@data\s*$',src.read_text(),flags=re.I|re.M)
        fields=attrs(h)
        records=[dict(zip(fields,r)) for r in csv.reader(x for x in d.splitlines() if x.strip())]
        require(len(set(names)&set(fields)-{'MIC'})==6,'Historical mapping differs')
        arff(dst,header,([r.get(k,'?') if k!='MIC' else '?' for k in names] for r in records))
        return len(records)
    training=out/'historical_training.arff'
    historical=out/'historical_replay.arff'
    panel=out/'historical_panel.arff'
    arff(training,header,([r[k] or '?' for k in names] for r in train))
    n=build(out/'original_without_ID.arff',historical)
    m=build(out/'batch5kleb60_tetramers.arff',panel)
    expected=[float(x) for x in re.findall(r'^\s*\d+\s+\?\s+([0-9.eE+-]+)\s+\?',Path('inputs/resultado_pred_kleb5.txt').read_text(),re.M)]
    require(n==len(expected)==211 and m==5,'Unexpected rows')
    if jar is None:
        print('Verified preparation: 527 training rows, 211 replay rows, 5 panel rows, 6 mapped descriptors')
        return
    # Check order and completeness before attributing any panel result to the historical model.
    cmd=['java','-cp',str(out/'classes')+':'+jar,'HistoricalRF',str(training),str(historical),str(panel)]
    sp.run(['javac','-cp',jar,'-d',str(out/'classes'),'HistoricalRF.java'],check=True)
    values=list(map(float,sp.check_output(cmd,universal_newlines=True).splitlines()))
    require(len(values)==n+m and all(map(math.isfinite,values)),'Invalid WEKA output')
    err=max(abs(a-b) for a,b in zip(expected,values[:n]))
    save(out/'historical_replay_validation.csv',[dict(n=n,max_abs_error=err,passed=err<=0.00050001)])
    require(err<=0.00050001,('Historical replay mismatch',err))
    current=read(out/'predictions.csv')
    ids=read(out/'batch5kleb60_row_map.csv')
    require([r['peptide_id'] for r in current]==[r['peptide_id'] for r in ids],'Panel order mismatch')
    save(out/'panel_comparison.csv',[dict(r,rf_historical_plus4_pool60=v) for r,v in zip(current,values[n:])])
    obs=[float(r['MIC_tetramer_uM']) for r in current]
    ranking=[]
    for label,pred in [('correct_plus5', [float(r['rf_output']) for r in current]),('historical_plus4_pool60',values[n:])]:
        p=[(obs[i]-obs[j])*(pred[i]-pred[j]) for i in range(m) for j in range(i) if max(obs[i],obs[j])/min(obs[i],obs[j])>2]
        ranking.append(dict(model=label,n=m,pairs_MIC_ratio_gt_2=len(p),concordant=sum(v>0 for v in p),discordant=sum(v<0 for v in p),ties_pred=sum(v==0 for v in p)))
    save(out/'ranking_comparison.csv',ranking)

if __name__=='__main__': main(Path(sys.argv[1]).resolve(),sys.argv[2] if len(sys.argv)>2 else None)
