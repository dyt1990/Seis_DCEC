# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:27:10 2018

@author: hp
"""

from Seismic_Conv1D_dec import DeepEmbeddingClustering
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
# Parameters
Method = 'DCEC'
filename = 'data'  # dim(Sample, lenth(data))
############ 导入聚类数据 ###############
file = h5py.File('F:\\dataset\\'+filename+'.mat','r')
train_orgin = np.float32(np.transpose(file['feature'][:]))
coor = np.float32(np.transpose(file['COOR'][:]))

sampleNo=len(train_orgin)
##COOR(:,1:6): Inline, Corssline, Coor_x, Coor_y
dataDim=np.shape(train_orgin)


Batch_size = 400
NumClusters = 7
Learning_rate = 0.01

# Parameters
Conv_filters = [16, 32, 64]        # Kernel number per layer
#Conv_filters = [8, 16, 32]        # Kernel number per layer
#Conv_filters = [32, 16, 8]        # Kernel number per layer
#Conv_filters = [16, 16, 32]        # Kernel number per layer
Kernel_size = 12        #卷积核尺寸
LatentSpace_Z = 25      #特征空间维度
Epochs = 5
InputDim = 64

lossname=(filename+'-'+Method+'-Cluster%d'%NumClusters+'-Z%d'%LatentSpace_Z+'-KerS%d'%Kernel_size)
# Check the layers' number
if np.int32(InputDim / (2**len(np.array(Conv_filters)))) <= 0:
    print('Conv_filters parmeter is out range of layers number !!!')
    exit()

# Adjust the dimensions of input data
# [samp_num, feat_dim]=train.shape
train_norm = np.zeros([dataDim[0], InputDim])
train_norm[:,:dataDim[1]] = train_orgin[:, :dataDim[1]]
X_train=np.reshape(train_norm, [train_norm.shape[0], train_norm.shape[1], 1])

############ DCEC 算法计算 ###############
DCEC = DeepEmbeddingClustering(n_clusters = NumClusters, input_dim = InputDim, 
                               learning_rate = Learning_rate, batch_size = Batch_size,
                               conv_filters = Conv_filters, kernel_size = Kernel_size,
                               LatentSpace_Z = LatentSpace_Z, finetune_epochs = Epochs)

clust_res_init, encoded_Z = DCEC.initialize(X_train)
#encoded_Z=DCEC.encoder.predict(X_train)
clust_res = DCEC.cluster(X_train, y=None, iter_max=2000, loss_name=lossname)

end_time = time.time()
time= end_time-start_time
print('time: ', time)
############ the clustering results ###############
Axis_format='In_Cro'

if Axis_format == 'In_Cro':
    result = np.zeros([sampleNo,3])
    result[:,0] = coor[:,1]     # Crossline
    result[:,1] = coor[:,0]     # Inline
    result[:,-1] = clust_res
elif Axis_format == 'Coor':
    result = np.zeros([sampleNo,3])
    result[:,0:-1] = coor[:,2:4]
    result[:,-1] = clust_res

#TODO clust_res_init
result_init = np.zeros([sampleNo,3])
result_init[:,0] = coor[:,1]     # Crossline
result_init[:,1] = coor[:,0]     # Inline
result_init[:,-1] = clust_res_init


################ Plot cluster result ##################
fig = plt.figure()
ax = plt.gca()
plt.scatter(result[:, 0], result[:, 1], s=2, c=result_init[:,2], cmap=plt.cm.get_cmap("jet", NumClusters))
plt.colorbar(ticks=range(NumClusters))
ax.invert_yaxis()

################# Save cluster result ##################
name=(filename+'-'+Method+'-'+Axis_format+'-Cluster%d'%NumClusters+'-Z%d'%LatentSpace_Z+'-KerS%d'%Kernel_size)
plt.savefig(name,dpi=600)
np.savetxt(name+'.txt', result, fmt='%0.2f %0.2f %d')
np.savetxt(name+'_init.txt', result_init, fmt='%0.2f %0.2f %d')
