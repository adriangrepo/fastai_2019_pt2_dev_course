#!/usr/bin/env python
# coding: utf-8






# ## Does nn.Conv2d init work well?



#export
from utils.nb_functions import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): 
    return (x-m)/s








x_train,y_train,x_valid,y_valid = get_data()
train_mean,train_std = x_train.mean(),x_train.std()
x_train = normalize(x_train, train_mean, train_std)
x_valid = normalize(x_valid, train_mean, train_std)




x_train = x_train.view(-1,1,28,28)
x_valid = x_valid.view(-1,1,28,28)
x_train.shape,x_valid.shape




n,*_ = x_train.shape
c = y_train.max()+1
nh = 32
n,c, _




l1 = nn.Conv2d(1, nh, 7)




x = x_valid[:100]




x.shape




def stats(x): 
    return x.mean(),x.std()




l1.weight.shape




stats(l1.weight),stats(l1.bias)




t = l1(x)




stats(t)




init.kaiming_normal_(l1.weight, a=1.)
stats(l1(x))




import torch.nn.functional as F




def f1(x,a=0): 
    return F.leaky_relu(l1(x),a)




init.kaiming_normal_(l1.weight, a=0)
stats(f1(x))




l1 = nn.Conv2d(1, nh, 7)
stats(f1(x))




l1.weight.shape




# receptive field size
rec_fs = l1.weight[0,0].numel()
rec_fs




nf,ni,*_ = l1.weight.shape
nf,ni




fan_in  = ni*rec_fs
fan_out = nf*rec_fs
fan_in,fan_out




def gain(a): 
    return math.sqrt(2.0 / (1 + a**2))




gain(1),gain(0),gain(0.01),gain(0.1),gain(math.sqrt(5.))




torch.zeros(10000).uniform_(-1,1).std()




1/math.sqrt(3.)




def kaiming2(x,a, use_fan_out=False):
    nf,ni,*_ = x.shape
    rec_fs = x[0,0].shape.numel()
    if use_fan_out:
        fan = nf*rec_fs 
    else:
        fan = ni*rec_fs
    std = gain(a) / math.sqrt(fan)
    bound = math.sqrt(3.) * std
    x.data.uniform_(-bound,bound)




kaiming2(l1.weight, a=0);
stats(f1(x))




kaiming2(l1.weight, a=math.sqrt(5.))
stats(f1(x))




class Flatten(nn.Module):
    def forward(self,x): 
        return x.view(-1)




m = nn.Sequential(
    nn.Conv2d(1,8, 5,stride=2,padding=2), nn.ReLU(),
    nn.Conv2d(8,16,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(32,1,3,stride=2,padding=1),
    nn.AdaptiveAvgPool2d(1),
    Flatten(),
)




y = y_valid[:100].float()




t = m(x)
stats(t)




l = mse(t,y)
l.backward()




stats(m[0].weight.grad)








for l in m:
    if isinstance(l,nn.Conv2d):
        init.kaiming_uniform_(l.weight)
        l.bias.data.zero_()




t = m(x)
stats(t)




l = mse(t,y)
l.backward()
stats(m[0].weight.grad)


# ## Export









