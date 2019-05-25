#!/usr/bin/env python
# coding: utf-8

# ## Matrix multiplication from foundations

# The *foundations* we'll assume throughout this course are:
# 
# - Python
# - Python modules (non-DL)
# - pytorch indexable tensor, and tensor creation (including RNGs - random number generators)
# - fastai.datasets

# ## Check imports








#export
from utils.nb_functions import *
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')




test_eq(TEST,'test')




# To run tests in console:
# ! python run_notebook.py 01_matmul.ipynb


# ## Get data



#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'




path = datasets.download_data(MNIST_URL, ext='.gz'); path




with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')




x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()




assert n==y_train.shape[0]==50000
test_eq(c,28*28)
test_eq(y_train.min(),0)
test_eq(y_train.max(),9)




mpl.rcParams['image.cmap'] = 'gray'




img = x_train[0]




img.view(28,28).type()




plt.imshow(img.view((28,28)));


# ## Initial python model



weights = torch.randn(784,10)




bias = torch.zeros(10)


# #### Matrix multiplication



def matmul(a,b):
    print(f'a.shape: {a.shape},b.shape: {b.shape}')
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c




m1 = x_valid[:5]
m2 = weights




m1.shape,m2.shape








t1.shape


# This is kinda slow - what if we could speed it up by 50,000 times? Let's try!



len(x_train)


# #### Elementwise ops

# Operators (+,-,\*,/,>,<,==) are usually element-wise.
# 
# Examples of element-wise operations:



a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a,b




a + b




(a < b).float().mean()




m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]); m


# Frobenius norm:
# 
# $$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$
# 
# *Hint*: you don't normally need to write equations in LaTeX yourself, instead, you can click 'edit' in Wikipedia and copy the LaTeX from there (which is what I did for the above equation). Or on arxiv.org, click "Download: Other formats" in the top right, then "Download source"; rename the downloaded file to end in `.tgz` if it doesn't already, and you should find the source there, including the equations to copy and paste.



(m*m).sum().sqrt()


# #### Elementwise matmul



def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c








890.1/5




#export
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)




test_near(t1,matmul(m1, m2))


# ### Broadcasting

# The term **broadcasting** describes how arrays with different shapes are treated during arithmetic operations.  The term broadcasting was first used by Numpy.
# 
# From the [Numpy Documentation](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html):
# 
#     The term broadcasting describes how numpy treats arrays with 
#     different shapes during arithmetic operations. Subject to certain 
#     constraints, the smaller array is “broadcast” across the larger 
#     array so that they have compatible shapes. Broadcasting provides a 
#     means of vectorizing array operations so that looping occurs in C
#     instead of Python. It does this without making needless copies of 
#     data and usually leads to efficient algorithm implementations.
#     
# In addition to the efficiency of broadcasting, it allows developers to write less code, which typically leads to fewer errors.
# 
# *This section was adapted from [Chapter 4](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression) of the fast.ai [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra) course.*

# #### Broadcasting with a scalar



a




a > 0


# How are we able to do a > 0?  0 is being **broadcast** to have the same dimensions as a.
# 
# For instance you can normalize our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar), using broadcasting.
# 
# Other examples of broadcasting with a scalar:



a + 1




m




2*m


# #### Broadcasting a vector to a matrix

# We can also broadcast a vector to a matrix:



c = tensor([10.,20,30]); c




m




m.shape,c.shape




m + c




c + m


# We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.



t = c.expand_as(m)




t




m + t




t.storage()




t.stride(), t.shape




c.shape


# You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).



d=c.unsqueeze(0)




d.shape




c.unsqueeze(1)




c.unsqueeze(1).shape




m




c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape




c.shape, c[None].shape,c[:,None].shape


# You can always skip trailling ':'s. And '...' means '*all preceding dimensions*'



c[None].shape,c[...,None].shape




c[:,None].expand_as(m)




m + c[:,None]




c[:,None]


# #### Matmul with broadcasting



def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c








885000/277




test_near(t1, matmul(m1, m2))


# #### Broadcasting Rules



c.shape




c




c[None,:]




c[None,:].shape




c[:,None]




c[:,None].shape




c[None,:] * c[:,None]




c[None] > c[:,None]


# When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when
# 
# - they are equal, or
# - one of them is 1, in which case that dimension is broadcasted to make it the same size
# 
# Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:
# 
#     Image  (3d array): 256 x 256 x 3
#     Scale  (1d array):             3
#     Result (3d array): 256 x 256 x 3
# 
# The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.

# ### Einstein summation

# Einstein summation (`einsum`) is a compact representation for combining products and sums in a general way. From the numpy docs:
# 
# "The subscripts string is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding operand. Whenever a label is repeated it is summed, so `np.einsum('i,i', a, b)` is equivalent to `np.inner(a,b)`. If a label appears only once, it is not summed, so `np.einsum('i', a)` produces a view of a with no changes."



# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)








885000/55




test_near(t1, matmul(m1, m2))


# ### pytorch op

# We can use pytorch's function or operator directly for matrix multiplication.







# time comparison vs pure python:
885000/18




t2 = m1@m2




test_near(t1, t2)




m1.shape,m2.shape




m3,m4  = m1.cuda(),m2.cuda()




t3 = m3.matmul(m4)






# ## Export



#!python notebook2script.py 01_matmul.ipynb











