import numpy as np
import pandas as pd

df = pd.read_csv('../sampleMatrixSelect100.txt', sep='\t', index_col=0)

map = np.loadtxt('./mart_export.txt', delimiter=',', dtype=str)
map = pd.Series(index=map[:,0], data=map[:,1])
# print(map)
cur_col = df.columns
