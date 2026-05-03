# Subpackage by Junho Kang

 # import common modules
from collections import Counter, defaultdict
import os, sys, glob, re
import scipy
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Diffxpy DEG
import anndata
import logging
import scipy.stats

def nmf_get_subset(idata, select,raw=True):
    if raw:
        adata = sc.AnnData(idata[select].raw.X)
        adata.var = idata.raw.var
    else:
        adata = sc.AnnData(idata[select].X)
        adata.var = idata.var
    adata.obs = idata.obs[select]
    adata.raw = adata.copy()
    sc.pp.highly_variable_genes(adata,subset=True)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata,n_neighbors=50)
    sc.tl.umap(adata)
    return adata


# Volcano Plot for DEG
def plot_volcano(query_dict, query_ct):
    query_dict[query_ct].plot_volcano(corrected_pval=True, min_fc=1.05, alpha=0.05, size=20)

# Summarizing DEG results
def deg_summary(query_dict, query_ct):
    deg_summ = query_dict[query_ct].summary()
    deg_summ = deg_summ[(deg_summ['pval']<0.01) & (deg_summ['log2fc']>1)].loc[:,['gene','pval','log2fc','mean']]
    return deg_summ

# Showing the statistics of Anndata.X
def ad_summary(ad):
    smp_summary = {}
    smp_summary['max'] = np.max(ad.X)
    smp_summary['mean'] = np.mean(ad.X)
    smp_summary['min'] = np.min(ad.X)
    smp_summary['len_obs'] = np.min(ad.shape[0])
    smp_summary['len_var'] = np.min(ad.shape[1])
    summ = pd.DataFrame.from_dict([smp_summary])
    print(summ)

# Creat volcano plot per patient from anndata
class sample_volcano():

    def __init__(self,adata,patient_id,anno_key,comp1,comp2,P=0.01,quick=True,
                 fc_cut_pval=0.5,n_pos_cell=10,n_patient_cell=10):
        '''
        param P :pseudocount for fc calculation
        '''
        from scipy.stats import ttest_ind, mannwhitneyu
        self.genelist = adata.raw.var_names
        
        # Removing Patients with cells less than n_patient_cell(10 by default)
        rmlist = []
        for f in Counter(adata.obs[patient_id]).items():
            if f[1]<n_patient_cell:
                rmlist.append(f[0])
            else:
                continue
        adata = adata[~adata.obs[patient_id].isin(rmlist)]
        
        adraw = adata.raw.to_adata()
        cond1 = adraw[adraw.obs[anno_key]==comp1]
        cond2 = adraw[adraw.obs[anno_key]==comp2]
        
        cond1 = pd.DataFrame(data=cond1.X.toarray(),index=cond1.obs[patient_id],columns=cond1.var_names)
        cond2 = pd.DataFrame(data=cond2.X.toarray(),index=cond2.obs[patient_id],columns=cond2.var_names)
        
        exp1 = cond1.groupby(cond1.index).mean()
        exp2 = cond2.groupby(cond2.index).mean()
        
        self.pval = []
        self.fc = []
        
        for i in adraw.var_names:
            # Calculating number of cell count with positive gene exp
            n_pos1 = np.sum(cond1[i]>0)
            n_pos2 = np.sum(cond2[i]>0)
            n_max = np.sum([n_pos1,n_pos2])
            
            # Calculating mean exp of each patient
            norm_count1 = np.mean(exp1[i])+P
            norm_count2 = np.mean(exp2[i])+P

            self.fc.append(np.log2(norm_count1/norm_count2))
            
            if quick:
                if np.abs(self.fc[-1])< fc_cut_pval:
                    self.pval.append(1)
                elif n_max < n_pos_cell:
                    self.pval.append(1)
                else:
                    self.pval.append(ttest_ind(exp1[i],exp2[i])[1])
            else:
                if np.abs(self.fc[-1])<0.00001:
                    self.pval.append(1)
                else:
                    self.pval.append(ttest_ind(exp1[i],exp2[i])[1])

        self.pval = np.array(self.pval)
        self.fc = np.array(self.fc)
            
    def draw(self, title=None,x_pos=1,pvalue_cut=1.5,to_show = 0.2,adjust_lim = 5,dotsize=8,
             show=True,sig_mode = 'auto',ylim=1.5,adjust=True,showlist=False):
        '''
        draw volcano plot
        param pvalue_cut :-log10Pvalue for cutoff
        sig_mode: ['auto','complex','pval']
        '''
        from adjustText import adjust_text
        plt.figure(figsize=(6,6))

        xpos = np.array(self.fc)
        ypos = -np.log10(np.array(self.pval))
        ypos[ypos==np.inf] = np.max(ypos[ypos!=np.inf])

        if sig_mode == 'complex':
            index = (np.abs(xpos))*ypos
            index_cut = np.percentile(index,100-to_show)
            sig = (np.abs(xpos) > x_pos) & (ypos > 2) & ((np.abs(xpos))*ypos > index_cut)
        elif sig_mode =='pval':
            sig = (np.abs(xpos) > x_pos) & (ypos > pvalue_cut)
        elif sig_mode =='pos_pval':
            sig = (xpos > x_pos) & (ypos > pvalue_cut)    
        elif sig_mode =='auto':
            index_cut = np.percentile(ypos,100-to_show)
            sig = (np.abs(xpos) > x_pos) & (ypos > index_cut)
        elif sig_mode =='manual':
            sig = np.array([True if x in showlist else False for x in self.genelist]) 
        else:
            print('error, check sig_mode')
            raise SystemError

        if title:
            plt.title(title,fontsize=14)
        sns.set_style('ticks')
        plt.xlabel('log2FoldChange',fontsize=12)
        plt.ylabel('-log10Pval',fontsize=12)
        plt.grid(False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(ylim,np.max(ypos)+0.3)
        plt.scatter(xpos,ypos,s=2, color='k', alpha =0.5, rasterized=True)
        plt.scatter(xpos[sig],ypos[sig],s=dotsize,color='red', rasterized=True)

        texts = []
        for i, gene in enumerate(self.genelist[sig]):
            texts.append(plt.text(xpos[sig][i],ypos[sig][i],gene,fontsize=8))
        
        if adjust:
            adjust_text(texts,only_move={'texts':'xy'},lim=adjust_lim)
        else:
            pass
        if show:
            plt.show()