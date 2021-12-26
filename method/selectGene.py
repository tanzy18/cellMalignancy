import numpy as np
import pandas as pd

NUM = 1000

if __name__ == '__main__':
    df = pd.read_csv('../sampleMatrixSelect100.txt',sep='\t',index_col=0)
    df = df.iloc[:,:-2] # 取出不含标签信息的数据
    v = df.var(axis=0)
    v_sort = v.sort_values(ascending=False, inplace=False)
    v_sort = v_sort[:NUM]
    selectGenes = np.array(v_sort.index)
    np.savetxt('selectGenes.txt', selectGenes, fmt='%s')