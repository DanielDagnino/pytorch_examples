import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

#------------------------------------------------------------------------------#
batch_size = 200

#------------------------------------------------------------------------------#
n_points = 20000
sigma = 0.5

points = np.zeros((n_points,2))
target = np.zeros((n_points,1))
for k in range(n_points):
	random = np.random.rand()
	if random<0.25:
		center = np.array([0,0])
		target[k,0] = 0
	elif random<0.5:
		center = np.array([2,2])
		target[k,0] = 1
	elif random<0.75:
		center = np.array([2,0])
		target[k,0] = 2
	else:
		center = np.array([0,2])
		target[k,0] = 3
	noise = np.random.randn(1,2)
	points[k,:] = center + sigma*noise

points_and_label = np.concatenate((points,target),axis=1)
points_and_label = pd.DataFrame(points_and_label)
points_and_label.to_csv('./data/clas.csv',index=False)

#------------------------------------------------------------------------------#
class mypoints(data.Dataset):
	def __init__(self, filename):
		pddata = pd.read_csv(filename).values
		self.target = pddata[:,2:]
		self.data = pddata[:,0:2]
		self.n_data = self.data.shape[0]
	
	def __len__(self):
		return self.n_data
	
	def __getitem__(self, index):
		return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])

mydata = mypoints('./data/clas.csv')

myloader = data.DataLoader(mydata,batch_size=batch_size,num_workers=0)

#------------------------------------------------------------------------------#
class mymodel(nn.Module):
	def __init__(self,n_in=2,n_out=4):
		super(mymodel,self).__init__()
		self.n_in  = n_in
		self.n_out = n_out
		
		self.linear = nn.Linear(self.n_in,self.n_out,bias=True)
		self.prob = nn.LogSoftmax(dim=1)
	
	def forward(self,x):
		x = self.linear(x)
		x = self.prob(x)
		return x

#------------------------------------------------------------------------------#
model = mymodel()

optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
criterium = nn.NLLLoss()

#------------------------------------------------------------------------------#
for iter, (data, target) in enumerate(myloader):
	data   = Variable(data,requires_grad=False)
	target = Variable(target.long(),requires_grad=False)
	
	optimizer.zero_grad()
	pred = model(data)
	loss = criterium(pred,target.view(-1))
	loss.backward()
	optimizer.step()
	
	if iter%10==0:
		print(loss.item())

#------------------------------------------------------------------------------#
target = target.numpy()
points = data.numpy()

select = target[:,0]==0
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],facecolors='b')

select = target[:,0]==1
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],facecolors='g')

select = target[:,0]==2
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],facecolors='tab:orange')

select = target[:,0]==3
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],facecolors='r')



pred = pred.exp().detach()
_, index = torch.max(pred,1)
pred = pred.numpy()
index = index.numpy()

select = index==0
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],s=60,marker='s',edgecolors='b',facecolors='none')

select = index==1
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],s=60,marker='s',edgecolors='g',facecolors='none')

select = index==2
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],s=60,marker='s',edgecolors='tab:orange',facecolors='none')

select = index==3
p0 = points[select,:]
plt.scatter(p0[:,0],p0[:,1],s=60,marker='s',edgecolors='r',facecolors='none')


plt.show()



