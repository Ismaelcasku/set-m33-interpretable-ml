import numpy as np
import pandas as pd

def summarize(group, nodes, deviation, variants):
    records = []
    nodes = nodes.copy()
    nodes['mic_pred'] = pd.to_numeric(nodes.mic_pred)
    nodes['pageranks'] = pd.to_numeric(nodes.pageranks)
    modules = sorted(nodes.groupby('modularity_class'),
                     key=lambda item: (item[1].mic_pred.mean(), item[0]))
    for rank, (module, members) in enumerate(modules, 1):
        for name, cols in variants.items():
            if not cols or not np.isfinite(deviation.loc[members.index, cols].to_numpy()).all():
                raise ValueError((group, module, name, 'Invalid descriptors'))
            similarity = -deviation.loc[members.index, cols].mean(axis=1)
            rho = (members.pageranks.corr(similarity, method='spearman')
                   if len(members) >= 3 and members.pageranks.nunique() > 1
                   and similarity.nunique() > 1 else np.nan)
            records.append(dict(group=group, module=module, mic_rank=rank,
                                n=len(members), mean_mic=members.mic_pred.mean(),
                                descriptors=name, n_descriptors=len(cols), rho=rho))
    return records
