#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:25:42 2018

@author: galeanojav
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


bacteria = pd.read_excel("FQ 16S.xlsx")
bac_cor = bacteria.iloc[:,1:]
dicc_bacteria = pd.DataFrame.to_dict(bac_cor)
dicc_patien = pd.DataFrame.to_dict(bac_cor.transpose())
# correlations between patients and bacteria

corr_pear_bac = pd.DataFrame.corr(bac_cor)
corr_pear_pat = pd.DataFrame.corr(bacteria.transpose())
corr_spear_bac = pd.DataFrame.corr(bac_cor, method='spearman')
corr_spear_pat = pd.DataFrame.corr(bacteria.transpose(), method='spearman')


plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_pear_bac)
plt.title("Pearson correlation")
plt.set_cmap('coolwarm')
plt.subplot(122)
plt.pcolor(corr_spear_bac)
plt.title("Spearman correlation")
plt.colorbar()
plt.savefig("correlation_bacteria.pdf")

plt.figure(figsize=(10,10))
plt.pcolor(corr_pear_pat)
plt.colorbar()


x = StandardScaler().fit_transform(bac_cor)
y = StandardScaler().fit_transform(bac_cor.transpose())
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
prinComp = pca.fit_transform(y)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC 1', 'PC 2'])

principal2 = pd.DataFrame(data = prinComp
             , columns = ['PC 1', 'PC 2'])

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot(principalDf['PC 1'],principalDf['PC 2'],'b+')
plt.title('Patients')
plt.subplot(122)
plt.plot(principal2['PC 1'],principal2['PC 2'],'ro')
plt.title('Bacteria')
plt.savefig("PCA.pdf")
