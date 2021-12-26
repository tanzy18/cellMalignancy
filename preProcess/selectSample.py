from os import sep
import numpy as np
import pandas as pd

df = pd.read_csv('../sampleMatrixSelect400Equ.txt',sep='\t',index_col=0)
print(df.iloc[:,-2:])
row_num = df.shape[0]

SELECTNUM = 100
idx = np.random.choice(np.arange(row_num),size=SELECTNUM,replace=False)
dfSelect = df.iloc[idx,:]
dfSelect.to_csv('../sampleMatrixSelect'+str(SELECTNUM)+'.txt', sep='\t')

'''
SELECTNUM = 400
df_pos , df_neg = df[df['label'] == 1] , df[df['label'] == 0]
pos_num , neg_num = df_pos.shape[0] , df_neg.shape[0]
print(pos_num , neg_num)
pos_idx = np.random.choice(np.arange(pos_num),size=SELECTNUM,replace=False)
neg_idx = np.random.choice(np.arange(neg_num),size=SELECTNUM,replace=False)
df_pos_sel = df.iloc[pos_idx,:]
df_neg_sel = df.iloc[neg_idx,:]
dfSelect   = pd.concat([df_pos_sel,df_neg_sel],axis=0,ignore_index=False)
dfSelect.to_csv('../sampleMatrixSelect'+str(SELECTNUM)+'Equ.txt', sep='\t')
'''

