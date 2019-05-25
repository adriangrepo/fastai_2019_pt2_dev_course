#!/usr/bin/env python
# coding: utf-8



import torch


# ### Why you need a good init

# To understand why initialization is important in a neural net, we'll focus on the basic operation you have there: matrix multiplications. So let's just take a vector `x`, and a matrix `a` initialized randomly, then multiply them 100 times (as if we had 100 layers). 



x = torch.randn(512)
a = torch.randn(512,512)




for i in range(100): 
    x = a @ x




x.mean(),x.std()


# The problem you'll get with that is activation explosion: very soon, your activations will go to nan. We can even ask the loop to break when that first happens:



x = torch.randn(512)
a = torch.randn(512,512)




for i in range(100): 
    x = a @ x
    if x.std() != x.std(): 
        break




i


# It only takes around 30 multiplications! On the other hand, if you initialize your activations with a scale that is too low, then you'll get another problem:



x = torch.randn(512)
a = torch.randn(512,512) * 0.01




for i in range(100): 
    x = a @ x




x.mean(),x.std()


# Here, every activation vanished to 0. So to avoid that problem, people have come with several strategies to initialize their weight matrices, such as:
# - use a standard deviation that will make sure x and Ax have exactly the same scale
# - use an orthogonal matrix to initialize the weight (orthogonal matrices have the special property that they preserve the L2 norm, so x and Ax would have the same sum of squares in that case)
# - use [spectral normalization](https://arxiv.org/pdf/1802.05957.pdf) on the matrix A  (the spectral norm of A is the least possible number M such that `torch.norm(A@x) <= M*torch.norm(x)` so dividing A by this M insures you don't overflow. You can still vanish with this)

# ### The magic number for scaling

# Here we will focus on the first one, which is the Xavier initialization. It tells us that we should use a scale equal to `1/math.sqrt(n_in)` where `n_in` is the number of inputs of our matrix.



import math




x = torch.randn(512)
a = torch.randn(512,512) / math.sqrt(512)




for i in range(100): 
    x = a @ x




x.mean(),x.std()


# And indeed it works. Note that this magic number isn't very far from the 0.01 we had earlier.



1/ math.sqrt(512)


# But where does it come from? It's not that mysterious if you remember the definition of the matrix multiplication. When we do `y = a @ x`, the coefficients of `y` are defined by
# 
# $$y_{i} = a_{i,0} x_{0} + a_{i,1} x_{1} + \cdots + a_{i,n-1} x_{n-1} = \sum_{k=0}^{n-1} a_{i,k} x_{k}$$
# 
# or in code:
# ```
# y[i] = sum([c*d for c,d in zip(a[i], x)])
# ```
# 
# Now at the very beginning, our `x` vector has a mean of roughly 0. and a standard deviation of roughly 1. (since we picked it that way).



x = torch.randn(512)
x.mean(), x.std()


# NB: This is why it's extremely important to normalize your inputs in Deep Learning, the initialization rules have been designed with inputs that have a mean 0. and a standard deviation of 1.
# 
# If you need a refresher from your statistics course, the mean is the sum of all the elements divided by the number of elements (a basic average). The standard deviation shows whether the data points stay close to the mean or are far away from it. It's computed by the following formula:
# 
# $$\sigma = \sqrt{\frac{1}{n}\left[(x_{0}-m)^{2} + (x_{1}-m)^{2} + \cdots + (x_{n-1}-m)^{2}\right]}$$
# 
# where m is the mean and $\sigma$ (the greek letter sigma) is the standard deviation. To avoid that square root, we also often consider a quantity called the variance, which is $\sigma$ squared. 
# 
# Here we have a mean of 0, so the variance is just the mean of x squared, and the standard deviation is its square root.
# 
# If we go back to `y = a @ x` and assume that we chose weights for `a` that also have a mean of 0, we can compute the variance of `y` quite easily. Since it's random, and we may fall on bad numbers, we repeat the operation 100 times.



mean,sqr = 0.,0.
for i in range(100):
    x = torch.randn(512)
    a = torch.randn(512, 512)
    y = a @ x
    mean += y.mean().item()
    sqr  += y.pow(2).mean().item()
mean/100,sqr/100


# Now that looks very close to the dimension of our matrix 512. And that's no coincidence! When you compute y, you sum 512 product of one element of a by one element of x. So what's the mean and the standard deviation of such a product of one element of `a` by one element of `x`? We can show mathematically that as long as the elements in `a` and the elements in `x` are independent, the mean is 0 and the std is 1.
# 
# This can also be seen experimentally:



mean,sqr = 0.,0.
for i in range(10000):
    x = torch.randn(1)
    a = torch.randn(1)
    y = a*x
    mean += y.item()
    sqr  += y.pow(2).item()
mean/10000,math.sqrt(sqr/10000)


# Then we sum 512 of those things that have a mean of zero, and a variance of 1, so we get something that has a mean of 0, and variance of 512. To go to the standard deviation, we have to add a square root, hence `math.sqrt(512)` being our magic number.
# 
# If we scale the weights of the matrix `a` and divide them by this `math.sqrt(512)`, it will give us a `y` of scale 1, and repeating the product as many times as we want and it won't overflow or vanish.

# ### Adding ReLU in the mix

# We can reproduce the previous experiment with a ReLU, to see that this time, the mean shifts and the variance becomes 0.5. This time the magic number will be `math.sqrt(2/512)` to properly scale the weights of the matrix.



mean,sqr = 0.,0.
for i in range(1000):
    x = torch.randn(1)
    a = torch.randn(1)
    y = a*x
    #item gets the float value out of the tensor
    y = 0 if y < 0 else y.item()
    mean += y
    sqr  += y ** 2
mean/1000,sqr/1000


# We can double check by running the experiment on the whole matrix product. The variance becomes 512/2 this time:



mean,sqr = 0.,0.
for i in range(100):
    x = torch.randn(512)
    a = torch.randn(512, 512)
    y = a @ x
    y = y.clamp(min=0)
    mean += y.mean().item()
    sqr  += y.pow(2).mean().item()
mean/100,sqr/100


# Or that scaling the coefficient with the magic number gives us a scale of 1.



mean,sqr = 0.,0.
for i in range(100):
    x = torch.randn(512)
    a = torch.randn(512, 512) * math.sqrt(2/512)
    y = a @ x
    y = y.clamp(min=0)
    mean += y.mean().item()
    sqr  += y.pow(2).mean().item()
mean/100,math.sqrt(sqr/100)


# The math behind is a tiny bit more complex, and you can find everything in the [Kaiming](https://arxiv.org/abs/1502.01852) and the [Xavier](http://proceedings.mlr.press/v9/glorot10a.html) paper but this gives the intuition behind those results.
