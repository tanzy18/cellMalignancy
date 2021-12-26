import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODE = 2
train_src,train_typ,train_tag,test_src,test_typ,test_tag = utils.split(mode=MODE)

N = 2
TYPE = np.unique(train_typ.values)
# mdl = PCA(n_components=N)
mdl = TSNE(n_components=N)
dcp_train_src = mdl.fit_transform(train_src)
print('Decompsition finished!')
'''
pos_idx , neg_idx = train_tag.values == 1 , train_tag.values == 0
pos_dcp_train_src = dcp_train_src[pos_idx,:]
neg_dcp_train_src = dcp_train_src[neg_idx,:]

pos_train_typ = train_typ[pos_idx]
neg_train_typ = train_typ[neg_idx]

plt.figure(1)
for item in TYPE:
    p_idx = pos_train_typ == item
    n_idx = neg_train_typ == item
    plt.scatter(pos_dcp_train_src[p_idx,0],pos_dcp_train_src[p_idx,1],s=3,label='pos'+item)
    plt.scatter(neg_dcp_train_src[n_idx,0],neg_dcp_train_src[n_idx,1],s=3,label='neg'+item)
'''
'''
cnt = 0
plt.figure(1)
for item in TYPE:
    idx = train_typ == item
    plt.scatter(dcp_train_src[idx,0],dcp_train_src[idx,1],s=3,label=item,c=plt.get_cmap('tab20')(cnt))
    cnt += 1
'''
TP = 'TCGA-KIRP'
pos_idx , neg_idx = train_tag.values == 1 , train_tag.values == 0
pos_dcp_train_src = dcp_train_src[pos_idx,:]
neg_dcp_train_src = dcp_train_src[neg_idx,:]
pos_train_typ = train_typ[pos_idx]
neg_train_typ = train_typ[neg_idx]
p_idx = pos_train_typ == TP
n_idx = neg_train_typ == TP
plt.figure(1)
plt.scatter(pos_dcp_train_src[p_idx,0],pos_dcp_train_src[p_idx,1],s=5,label='pos'+TP,c='r')
plt.scatter(neg_dcp_train_src[n_idx,0],neg_dcp_train_src[n_idx,1],s=5,label='neg'+TP,c='b')

plt.legend()
plt.savefig('./decompResult/KIRP.png')
plt.show()