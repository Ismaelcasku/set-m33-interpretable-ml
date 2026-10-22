from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

SPECIES=[('ecoli',r'$\it{E.\ coli}$ TG1',[4,5,6,7]),('paeruginosa',r'$\it{P.\ aeruginosa}$ PAO-1',[4,5,6,7]),('kpneumoniae',r'$\it{K.\ pneumoniae}$ ATCC 13833',[4,5,6])]

def importance(directory):
    values=np.load(directory/'shap_values.npy',mmap_mode='r')
    names=json.loads((directory/'features.json').read_text())
    mean=np.mean(np.abs(np.asarray(values,dtype=float)),axis=0)
    if not np.isfinite(mean).all() or mean.sum()<=0:raise ValueError(f'Invalid SHAP importance: {directory}')
    return pd.Series(mean/mean.sum(),index=names)

def save(fig,stem):
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'figures');parser.add_argument('--ecoli-only',action='store_true');args=parser.parse_args()
    r=args.results_root;out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    selected=SPECIES[:1] if args.ecoli_only else SPECIES
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    matrices=[];full=[]
    for species,title,charges in selected:
        matrix=pd.DataFrame({f'+{c}':importance(r/f'shap_reference/{species}_{c}plus') for c in charges}).fillna(0)
        matrix.index.name='descriptor';matrix.to_csv(out/f'{species}_relative_SHAP_all.csv')
        top=matrix.mean(axis=1).sort_values(ascending=False,kind='stable').head(10).index
        matrices.append(matrix.loc[top]);full.append(matrix)
        matrix.loc[top].to_csv(out/f'{species}_Figure2_values.csv')
    vmax=max(float(m.to_numpy().max()) for m in matrices)
    fig,axes=plt.subplots(len(selected),1,figsize=(8.4,4.8*len(selected)),squeeze=False)
    fig.subplots_adjust(left=.43,right=.85,bottom=.05,top=.96,hspace=.43)
    for i,((species,title,charges),matrix) in enumerate(zip(selected,matrices)):
        ax=axes[i,0];im=ax.imshow(matrix.to_numpy(),cmap='magma',vmin=0,vmax=vmax,aspect='auto')
        ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=9)
        ax.set_xticks(np.arange(len(charges)),[f'+{c}' for c in charges]);ax.tick_params(length=0,pad=8)
        ax.set(xlabel='Charge',title=title)
        ax.text(-.16,1.08,chr(65+i),transform=ax.transAxes,fontsize=19,fontweight='bold')
        for row in range(len(matrix)):
            for col in range(len(charges)):
                value=matrix.iloc[row,col];ax.text(col,row,f'{value:.2f}',ha='center',va='center',fontsize=8,color='black' if value/vmax>.65 else 'white')
        for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.89,.22,.024,.56]);fig.colorbar(im,cax=cax,label='Relative mean |SHAP|')
    save(fig,out/'Figure_2_SHAP')

    fig=plt.figure(figsize=(11,10) if len(selected)>1 else (6.5,5.5))
    if len(selected)>1:
        grid=fig.add_gridspec(2,4,left=.065,right=.88,bottom=.06,top=.95,wspace=.65,hspace=.3)
        axes=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:]),fig.add_subplot(grid[1,1:3])]
    else:
        axes=[fig.add_subplot(111)];fig.subplots_adjust(right=.82)
    cmap=plt.get_cmap('viridis');norm=Normalize(4,7)
    for i,((species,title,charges),ax) in enumerate(zip(selected,axes)):
        base=r/('historical_umap' if (r/'historical_umap').is_dir() else 'primary_umap')/species
        emb_path=base/'historical_embedding.csv'
        if not emb_path.exists():emb_path=base/'primary_embedding.csv'
        emb=pd.read_csv(emb_path)
        if not np.isfinite(emb[['UMAP1','UMAP2']]).all().all():raise ValueError('Nonfinite embedding')
        # Preserve stored row order and coordinates; do not refit UMAP here.
        ax.scatter(emb.UMAP1,emb.UMAP2,c=emb.cationic_charge,cmap=cmap,norm=norm,s=3,linewidths=0,rasterized=False)
        ax.set(title=title,xlabel='UMAP-1',ylabel='UMAP-2');ax.set_xticks([]);ax.set_yticks([])
        ax.text(-.075,1.06,chr(65+i),transform=ax.transAxes,fontsize=19,fontweight='bold')
    cax=fig.add_axes([.92,.2,.018,.58]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[4,5,6,7],label='Charge (+)')
    save(fig,out/'Figure_3_UMAP')

    # Supplementary top-20 panels: same within-model normalization as Figure 2.
    detailed=out/'S4_individual';detailed.mkdir(exist_ok=True)
    with PdfPages(out/'Figure_S4_SHAP_top20_all.pdf') as pdf:
        for species,title,charges in selected:
            for charge in charges:
                top=importance(r/f'shap_reference/{species}_{charge}plus').nlargest(20).sort_values()
                fig,ax=plt.subplots(figsize=(9,6.8));fig.subplots_adjust(left=.48,right=.96,bottom=.12,top=.91)
                ax.barh(np.arange(len(top)),top.to_numpy(),color=plt.get_cmap('viridis')(norm(charge)))
                ax.set_yticks(np.arange(len(top)),top.index,fontsize=9);ax.set(xlabel='Relative mean |SHAP|',title=f'{title}, +{charge}')
                ax.spines[['top','right']].set_visible(False);pdf.savefig(fig,bbox_inches='tight')
                save(fig,detailed/f'{species}_{charge}plus_top20')
    (out/'FIGURE_SCOPE.txt').write_text('Figure 2: top ten descriptors ranked by mean relative absolute SHAP across charges; missing selected-pool descriptors shown as zero. Figure 3: stored primary Euclidean UMAP, no refit or coordinate alignment. E. coli updated; other organisms reused from copied archived outputs. S4: eleven individual top-20 panels. Label design follows the supplied manuscript; review final rendering before insertion.\n')
    print(f'Figures completed in {out}',flush=True)

if __name__=='__main__':main()
