#!/usr/bin/env python
# coding: utf-8








import torch
import matplotlib.pyplot as plt


# ## Callbacks

# ### Callbacks as GUI events



import ipywidgets as widgets




def f(o): 
    print('hi')


# From the [ipywidget docs](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html):
# 
# - *the button widget is used to handle mouse clicks. The on_click method of the Button can be used to register function to be called when the button is clicked*



w = widgets.Button(description='Click me')




w




w.on_click(f)


# *NB: When callbacks are used in this way they are often called "events".*
# 
# Did you know what you can create interactive apps in Jupyter with these widgets? Here's an example from [plotly](https://plot.ly/python/widget-app/):
# 
# ![](https://cloud.githubusercontent.com/assets/12302455/16637308/4e476280-43ac-11e6-9fd3-ada2c9506ee1.gif)

# ### Creating your own callback



from time import sleep




def slow_calculation():
    res = 0
    for i in range(5):
        res += i*i
        sleep(1)
    return res




slow_calculation()




def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        res += i*i
        sleep(1)
        if cb: cb(i)
    return res




def show_progress(epoch):
    print(f"Awesome! We've finished epoch {epoch}!")




slow_calculation(show_progress)


# ### Lambdas and partials



slow_calculation(lambda o: print(f"Awesome! We've finished epoch {o}!"))




def show_progress(exclamation, epoch):
    print(f"{exclamation}! We've finished epoch {epoch}!")




slow_calculation(lambda o: show_progress("OK I guess", o))




def make_show_progress(exclamation):
    _inner = lambda epoch: print(f"{exclamation}! We've finished epoch {epoch}!")
    return _inner




slow_calculation(make_show_progress("Nice!"))




def make_show_progress(exclamation):
    # Leading "_" is generally understood to be "private"
    def _inner(epoch): 
        print(f"{exclamation}! We've finished epoch {epoch}!")
    return _inner




slow_calculation(make_show_progress("Nice!"))




f2 = make_show_progress("Terrific")




slow_calculation(f2)




slow_calculation(make_show_progress("Amazing"))




from functools import partial




slow_calculation(partial(show_progress, "OK I guess"))




f2 = partial(show_progress, "OK I guess")


# ### Callbacks as callable classes



class ProgressShowingCallback():
    def __init__(self, exclamation="Awesome"): 
        print('__init__')
        self.exclamation = exclamation
    def __call__(self, epoch): 
        print('__call__')
        print(f"{self.exclamation}! We've finished epoch {epoch}!")




cb = ProgressShowingCallback("Just super")




slow_calculation(cb)


# ### Multiple callback funcs; `*args` and `**kwargs`



def f(*args, **kwargs): 
    print(f"args: {args}; kwargs: {kwargs}")




f(4, 'a', thing1="hello", thing2='thing2')


# NB: We've been guilty of over-using kwargs in fastai - it's very convenient for the developer, but is annoying for the end-user unless care is taken to ensure docs show all kwargs too. kwargs can also hide bugs (because it might not tell you about a typo in a param name). In [R](https://www.r-project.org/) there's a very similar issue (R uses `...` for the same thing), and matplotlib uses kwargs a lot too.



def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb: 
            cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb: 
            cb.after_calc(i, val=res)
    return res




class PrintStepCallback():
    def __init__(self): 
        pass
    def before_calc(self, *args, **kwargs): 
        print(f"About to start")
    def after_calc (self, *args, **kwargs): 
        print(f"Done step")




slow_calculation(PrintStepCallback())




class PrintStatusCallback():
    def __init__(self): 
        pass
    def before_calc(self, epoch, **kwargs): 
        print(f"About to start: {epoch}")
    def after_calc (self, epoch, val, **kwargs): 
        print(f"After {epoch}: {val}")




slow_calculation(PrintStatusCallback())


# ### Modifying behavior

# 
# hasattr(object, name)
# 
#     The arguments are an object and a string. The result is True if the string is the name of one of the object’s attributes, False if not. (This is implemented by calling getattr(object, name) and seeing whether it raises an AttributeError or not.)
# 



def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb and hasattr(cb,'before_calc'): 
            cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb and hasattr(cb,'after_calc'):
            if cb.after_calc(i, res):
                print("stopping early")
                break
    return res




class PrintAfterCallback():
    def after_calc (self, epoch, val):
        print(f"After {epoch}: {val}")
        if val>10: 
            return True




slow_calculation(PrintAfterCallback())




class SlowCalculator():
    def __init__(self, cb=None): 
        self.cb,self.res = cb,0
    
    def callback(self, cb_name, *args):
        if not self.cb: 
            return
        cb = getattr(self.cb,cb_name, None)
        if cb: 
            return cb(self, *args)

    def calc(self):
        print('SlowCalculator.calc()')
        for i in range(5):
            self.callback('before_calc', i)
            self.res += i*i
            sleep(1)
            if self.callback('after_calc', i):
                print("stopping early")
                break




class ModifyingCallback():
    def after_calc (self, calc, epoch):
        print('ModifyingCallback.after_calc()')
        print(f"After {epoch}: {calc.res}")
        if calc.res>10: 
            return True
        if calc.res<3: 
            calc.res = calc.res*2




calculator = SlowCalculator(ModifyingCallback())




calculator.calc()
calculator.res


# ## `__dunder__` thingies

# Anything that looks like `__this__` is, in some way, *special*. Python, or some library, can define some functions that they will call at certain documented times. For instance, when your class is setting up a new object, python will call `__init__`. These are defined as part of the python [data model](https://docs.python.org/3/reference/datamodel.html#object.__init__).
# 
# For instance, if python sees `+`, then it will call the special method `__add__`. If you try to display an object in Jupyter (or lots of other places in Python) it will call `__repr__`.



class SloppyAdder():
    def __init__(self,o): 
        self.o=o
    def __add__(self,b): 
        return SloppyAdder(self.o + b.o + 0.01)
    def __repr__(self): 
        return str(self.o)




a = SloppyAdder(1)
b = SloppyAdder(2)
a+b


# Special methods you should probably know about (see data model link above) are:
# 
# - `__getitem__`
# - `__getattr__`
# - `__setattr__`
# - `__del__`
# - `__init__`
# - `__new__`
# - `__enter__`
# - `__exit__`
# - `__len__`
# - `__repr__`
# - `__str__`

# ## Variance and stuff

# ### Variance

# Variance is the average of how far away each data point is from the mean. E.g.:



t = torch.tensor([1.,2.,4.,18])




t




m = t.mean(); m




(t-m).mean()




t-m


# Oops. We can't do that. Because by definition the positives and negatives cancel out. So we can fix that in one of (at least) two ways:



(t-m).pow(2).mean()




(t-m).abs().mean()


# But the first of these is now a totally different scale, since we squared. So let's undo that at the end.



(t-m).pow(2).mean().sqrt()


# They're still different. Why?
# 
# Note that we have one outlier (`18`). In the version where we square everything, it makes that much bigger than everything else.
# 
# `(t-m).pow(2).mean()` is referred to as **variance**. It's a measure of how spread out the data is, and is particularly sensitive to outliers.
# 
# When we take the sqrt of the variance, we get the **standard deviation**. Since it's on the same kind of scale as the original data, it's generally more interpretable. However, since `sqrt(1)==1`, it doesn't much matter which we use when talking about *unit variance* for initializing neural nets.
# 
# `(t-m).abs().mean()` is referred to as the **mean absolute deviation**. It isn't used nearly as much as it deserves to be, because mathematicians don't like how awkward it is to work with. But that shouldn't stop us, because we have computers and stuff.
# 
# Here's a useful thing to note about variance:



(t-m).pow(2).mean(), (t*t).mean() - (m*m)


# You can see why these are equal if you want to work thru the algebra. Or not.
# 
# But, what's important here is that the latter is generally much easier to work with. In particular, you only have to track two things: the sum of the data, and the sum of squares of the data. Whereas in the first form you actually have to go thru all the data twice (once to calculate the mean, once to calculate the differences).
# 
# Let's go steal the LaTeX from [Wikipedia](https://en.wikipedia.org/wiki/Variance):
# 
# $$\operatorname{E}\left[X^2 \right] - \operatorname{E}[X]^2$$

# E[X] is the expected value of X ie weighted average of all values of X-assuming each is equiprobable

# ### Covariance and correlation

# Here's how Wikipedia defines covariance:
# 
# $$\operatorname{cov}(X,Y) = \operatorname{E}{\big[(X - \operatorname{E}[X])(Y - \operatorname{E}[Y])\big]}$$



t


# Let's see that in code. So now we need two vectors.

# torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
# 
# Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. torch.randn_like(input) is equivalent to torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).



# `u` is twice `t`, plus a bit of randomness
u = t*2
u *= torch.randn_like(t)/10+0.95

plt.scatter(t, u);




prod = (t-t.mean())*(u-u.mean()); prod




prod.mean()




v = torch.randn_like(t)
plt.scatter(t, v);




((t-t.mean())*(v-v.mean())).mean()


# It's generally more conveniently defined like so:
# 
# $$\operatorname{E}\left[X Y\right] - \operatorname{E}\left[X\right] \operatorname{E}\left[Y\right]$$



cov = (t*v).mean() - t.mean()*v.mean(); cov


# From now on, you're not allowed to look at an equation (or especially type it in LaTeX) without also typing it in Python and actually calculating some values. Ideally, you should also plot some values.
# 
# Finally, here is the Pearson correlation coefficient:
# 
# $$\rho_{X,Y}= \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}$$



cov / (t.std() * v.std())


# It's just a scaled version of the same thing. Question: *Why is it scaled by standard deviation, and not by variance or mean or something else?*

# ## Softmax

# Here's our final `logsoftmax` definition:



def log_softmax(x): 
    return x - x.exp().sum(-1,keepdim=True).log()


# which is:
# 
# $$\hbox{logsoftmax(x)}_{i} = x_{i} - \log \sum_{j} e^{x_{j}}$$ 
# 
# And our cross entropy loss is:
# $$-\log(p_{i})$$

# ## Browsing source code

# - Jump to tag/symbol by with (with completions)
# - Jump to current tag
# - Jump to library tags
# - Go back
# - Search
# - Outlining / folding
