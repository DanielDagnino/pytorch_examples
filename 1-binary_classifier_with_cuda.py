import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

#------------------------------------------------------------------------------#
if torch.cuda.is_available():
	print('using CUDA')
	torch.cuda.empty_cache()
else:
	print('using CPU')

#------------------------------------------------------------------------------#
class binary_gen(data.Dataset):
	
	def __init__(self, sigma=1.0, alpha=3., bias=2., n_samples=10000):
		self.sigma = sigma
		self.alpha = alpha
		self.bias = bias
		self.n_samples = n_samples
	
	def __len__(self):
		return self.n_samples
	
	def __getitem__(self, index):
		r = self.sigma*np.random.randn(2,1).astype(np.float)
		if np.random.rand()>0.5:
			data = r + np.array([[0],[0]]).astype(np.float)
			target = [0.]
		else:
			data = r + np.array([[2],[2]]).astype(np.float)
			target = [1.]
		data = torch.Tensor(data).view(2)
		target = torch.Tensor(target)
		return data, target

#------------------------------------------------------------------------------#
data_set = binary_gen()
loader = data.DataLoader(
    data_set, batch_size=100, shuffle=False, drop_last=False,
    num_workers=2, pin_memory=True)

#------------------------------------------------------------------------------#
class binary_model(nn.Module):
	def __init__(self, n_in, n_out):
		super(binary_model,self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		
		self.lin = nn.Linear(n_in, n_out, bias=True)
		self.prob = nn.Sigmoid()
		
	def forward(self, x):
		x = self.lin(x)
		x = self.prob(x)
		return x

#------------------------------------------------------------------------------#
model = binary_model(2,1).cuda()
criterion = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#------------------------------------------------------------------------------#
for k, (data, target) in enumerate(loader):
	data   = Variable(data.cuda(), requires_grad=False)
	target = Variable(target.cuda(), requires_grad=False)
	
	optimizer.zero_grad()
	pred = model(data)
	
	loss = criterion(pred, target)
	loss.backward()
	optimizer.step()
	
	if k%1==0:
		print('loss {:.4f}'.format(loss))
		
#------------------------------------------------------------------------------#
my_gen = binary_gen(n_samples=100)
loader = torch.utils.data.DataLoader(
    data_set, batch_size=100, shuffle=False, drop_last=False,
    num_workers=0, pin_memory=True)

for k, (data, target) in enumerate(loader):
	data   = Variable(data.cuda(), requires_grad=False)
	target = Variable(target.cuda(), requires_grad=False)
	pred = model(data).detach().cpu().numpy()
	
	par = [par for par in model.parameters()]

#------------------------------------------------------------------------------#
import matplotlib.pyplot as plt

target = target.cpu().numpy()
data = data.cpu().numpy()

neg = (target==0).reshape(-1)
pos_neg_data = data[neg,:]
pos_pos_data = data[np.logical_not(neg),:]

neg = (pred>0.5).reshape(-1)
pos_neg_data2 = data[neg,:]
pos_pos_data2 = data[np.logical_not(neg),:]

alpha = par[0][0,0].item()
beta = par[0][0,1].item()
bias = par[1].item()

x = np.linspace(-2, 3, num=100)
y = (alpha*x + bias)/(-beta)

plt.plot(x,y)
plt.scatter(pos_neg_data[:,0], pos_neg_data[:,1], s=20)
plt.scatter(pos_pos_data[:,0], pos_pos_data[:,1], s=20)
plt.scatter(pos_neg_data2[:,0], pos_neg_data2[:,1], marker='s', s=50, edgecolors='r', facecolors='none')
plt.scatter(pos_pos_data2[:,0], pos_pos_data2[:,1], marker='s', s=50, edgecolors='b', facecolors='none')
plt.show()

#------------------------------------------------------------------------------#
print('End')

