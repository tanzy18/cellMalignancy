import os
import numpy as np
import pandas as pd
from tqdm import tqdm

mapping = {'Additional - New Primary':True,'Additional Metastatic':True,'Metastatic':True,
            'Primary Tumor':True,'Recurrent Tumor':True,'Solid Tissue Normal':False}
sampleSheet = pd.read_csv('./gdc_sample_sheet.2021-11-01.tsv',delimiter='\t',
                            usecols=['File ID','Project ID','Sample Type'],index_col=0)
file = '../TCGA/00a48a5d-b138-4504-a9ac-fb919cb81185/9e8b528b-1172-4c07-a09b-ebb23cf2310c.FPKM.txt.gz'
fileID = '00a48a5d-b138-4504-a9ac-fb919cb81185'
splArr = pd.read_csv(file,delimiter='\t',compression='gzip',index_col=0,
                names=[fileID])
splArr.loc['type']  = sampleSheet.loc[fileID,'Project ID']
splArr.loc['label'] = mapping[sampleSheet.loc[fileID,'Sample Type']]
# print(spl1)

# file = './TCGA/00a137ad-22a5-457f-93c5-c56cf902e3d4/62e6035d-ecca-4e98-a1c4-4663be031c56.FPKM.txt.gz'
# spl2 = pd.read_csv(file,delimiter='\t',compression='gzip',index_col=0,
#                 names=['00a48a5d-b138-4504-a9ac-fb919cb81185'])
# print(spl2)
# splArr = pd.concat((spl1,spl2),axis=1)
# splArr = spl1.join(spl2)
# print(splArr)
# cnt = 0
for fileID in tqdm(os.listdir('../TCGA')):
    if fileID == '00a48a5d-b138-4504-a9ac-fb919cb81185':
        continue
    for fileName in os.listdir('../TCGA/' + fileID):
        if fileName.endswith('.gz'):
            spl = pd.read_csv('../TCGA/' + fileID + '/' + fileName,delimiter='\t',compression='gzip',index_col=0,
                              names=[fileID])
            spl.loc['type']  = sampleSheet.loc[fileID,'Project ID']
            spl.loc['label'] = mapping[sampleSheet.loc[fileID,'Sample Type']]
            splArr = splArr.join(spl)
    #         cnt += 1
    # if cnt >= 30:
    #     break
# print(splArr)
splArr = splArr.T
splArr.to_csv('../sampleMatrix.txt',sep='\t')