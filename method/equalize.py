from imblearn.combine import SMOTEENN, SMOTETomek
# from imblearn.over_sampling import SVMSMOTE, SMOTE
from utils import split
import pandas as pd

def equalize(train_src, train_tag):
    sm = SMOTETomek()
    train_src_resample , train_tag_resample = sm.fit_resample(train_src, train_tag)
    return train_src_resample, train_tag_resample

if __name__ == '__main__':
    train_src,_,train_tag,_,_,_ = split(mode=1)
    print(sum(train_tag == True), sum(train_tag == False))
    sm = SMOTETomek()
    train_src_resample , train_tag_resample = sm.fit_resample(train_src, train_tag)
    print(train_src_resample.shape, train_tag_resample.shape)
    table = pd.concat([train_src_resample, train_tag_resample], axis=1)
    # table = table.T
    # table.to_csv('./resample.txt')
    print(sum(train_tag_resample == True), sum(train_tag_resample == False))