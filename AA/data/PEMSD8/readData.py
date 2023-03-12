# _*_ coding: utf-8 _*_
# @time     :2023/3/8 20:29
# @Author   :jc
# @File     :readData.py
import numpy as np
# ['x', 'y']
# (10699, 12, 170, 2)#速度  流量
# (10699, 12, 170, 2)
data=np.load('train.npz')
files = data.files
print(files)
print(data['x'].shape)
print(data['y'].shape)
print(data['x'][0,0,0,0])
print(data['x'][0,0,0,1])
print(data['y'][0,0,0,0])
print(data['y'][0,0,0,1])
data2=np.load('pems08.npz') #(17856, 170, 3) 流量  占有率  速度
print(data2.files)
print(data2['data'].shape)
print(data2['data'][0,0,0])
print(data2['data'][0,0,1])
print(data2['data'][0,0,2])
# print(data.shape())
print('------------------')
print(data['x'][1,0,0,0])
print(data['x'][1,0,0,1])
print(data2['data'][1,0,0])
print(data2['data'][1,0,2])