import pandas as pd

df = pd.read_csv('../sampleMatrixFull.txt',sep='\t',index_col=0)
print('read finished!')
df = df.T
print('transpose finished!')
df.to_csv('../sampleMatrixFullTransposed.txt',sep='\t')