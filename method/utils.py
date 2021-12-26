import numpy as np
import pandas as pd

def split(mode=1,ratio=0.95):
    if mode == 1:
        df = pd.read_csv('../sampleMatrixSelect100.txt',sep='\t',index_col=0)
    elif mode == 2:
        df = pd.read_csv('../sampleMatrixSelect400Equ.txt',sep='\t',index_col=0)
    elif mode == 3:
        df = pd.read_csv('../sampleMatrix.txt',sep='\t',index_col=0)
    else:
        pass
    print('Read finished!')
    row_num = df.shape[0]
    train_idx = np.random.choice(np.arange(row_num), size=int(row_num*ratio), replace=False)
    test_idx  = np.setdiff1d(np.arange(row_num), train_idx)
    train_src = df.iloc[train_idx,:-2]
    train_typ = df.iloc[train_idx,-2]
    train_tag = df.iloc[train_idx,-1]
    test_src  = df.iloc[test_idx,:-2]
    test_typ  = df.iloc[test_idx,-2]
    test_tag  = df.iloc[test_idx,-1]
    return train_src,train_typ,train_tag,test_src,test_typ,test_tag

def splitAfterStd(data,tag,ratio=0.75):
    row_num = data.shape[0]
    train_idx = np.random.choice(np.arange(row_num), size=int(row_num*ratio), replace=False)
    test_idx  = np.setdiff1d(np.arange(row_num), train_idx)
    train_data , train_tag = data.iloc[train_idx,:] , tag[train_idx]
    test_data  , test_tag  = data.iloc[test_idx,:] , tag[test_idx]
    return train_data, test_data, train_tag, test_tag


if __name__ == '__main__':
    train_src,train_typ,train_tag,test_src,test_typ,test_tag = split(mode=1)
    # df = pd.read_csv('../sampleMatrixSelect400Equ.txt',sep='\t',index_col=0)
    print(train_src.shape)