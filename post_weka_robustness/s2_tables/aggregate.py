from pathlib import Path
import argparse,json
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
root=a.work/'results'; tables=[]; reports=[]; md=[]
for i,sp in enumerate(['ecoli','paeruginosa','kpneumoniae'],1):
    report=json.loads((root/sp/'validation.json').read_text())
    if report['status']!='completed' or not report['inputs_unchanged'] or not report['pinned_environment']:
        raise RuntimeError(f'Validation failed for {sp}')
    reports.append(report);tables.append(pd.read_csv(root/sp/'table_full_precision.csv'))
    md.append(f'Table S2.{i}: {sp}\n\n'+(root/sp/'table.md').read_text())
(root/'Tables_S2_1_to_S2_3.md').write_text('\n\n'.join(md))
pd.concat(tables,ignore_index=True).to_csv(root/'Tables_S2_all_full_precision.csv',index=False,float_format='%.17g')
(root/'METHODS.md').write_text((a.work/'METHODS.md').read_text())
(root/'COMPLETE.json').write_text(json.dumps({'status':'completed','species':[r['species'] for r in reports],'all_inputs_unchanged':True},indent=2)+'\n')
print('All three replacement tables completed.')
