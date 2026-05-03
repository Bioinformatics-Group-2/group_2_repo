
import os, sys
import numpy as np
import scipy as scipy
import scanpy as sc
import pandas as pd
import glob
from sklearn.decomposition import NMF
from multiprocessing import Pool
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#Loading adata
adata=sc.read('../07_merged/CCA_v01.CCA_total_2211.h5ad')

try:
    adata=adata.raw.to_adata()
except:
    pass

try:
    adata.X=scipy.sparse.csr_matrix.toarray(adata.X)
except:
    pass

#Defining K parameters, cell type, and batch keey
n_components=[5,6,7,8,9]
celltype='FibroEndoPeri'
batchkey='Patient_Organ_Tissue'
path='../01_NMF_perpt'

#Subsetting adata to the cell type category of interest
adata=adata[adata.obs['anno_category']==celltype]
adata.raw=adata
print('{} subset complete'.format(celltype))

#Scaling
sc.pp.scale(adata,max_value=10)
print('scaling complete')

#Negative values to zero
ad={}
for f in set(adata.obs[batchkey]):
    sdata=adata[adata.obs[batchkey]==f]
    temp=np.where(sdata.X<0,0,sdata.X)
    ad[f]=temp
    del sdata,temp
   
#NMF analysis
def nmf(items): # Dictionary .items Input
    for comp in n_components:
        folder=path+'/'+'{}_{}components'.format(celltype,str(comp))
        try:
            os.mkdir(folder)
        except:
            pass
        print(items[0],str(comp))
        model = NMF(n_components=comp,max_iter=1500)
        W = model.fit_transform(items[1])
        H = model.components_
        df=pd.DataFrame(H,columns=adata.var_names).T
        df.to_csv('{}/{}comp_{}_all.csv'.format(folder,str(comp),items[0]))

if __name__=="__main__":
    pool = Pool(processes = 20)
    pool.map(nmf,ad.items())
    pool.close()
    pool.join()

