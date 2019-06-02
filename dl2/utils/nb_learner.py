
from utils.nb_functions import *
from utils.nb_classes_l10_revised import *
from utils.nb_classes_cnn import *
from utils.nb_optimizer import *

class Learner():
    def __init__(self, model, data, loss_func, opt=None, opt_func=sgd_opt, lr=None,
                 cbs=None, cb_funcs=None):
        print('>>Learner.__init__()')
        assert opt or lr
        if not opt:
            opt = opt_func(model.parameters(), lr=lr)
        self.opt = opt
        self.model,self.data,self.loss_func,self.opt = model,data,loss_func,opt
        self.in_train,self.logger = False,print

        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train:
                return
            self.loss.backward()
            self('after_backward')
            #temp = self.opt.func
            #temp.step()
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i,(xb,yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def do_begin_fit(self, epochs):
        self.epochs,self.loss = epochs,tensor(0.)
        for cb in self.cbs:
            cb.set_runner(self)
        self('begin_fit')

    def do_begin_epoch(self, epoch):
        self.epoch,self.dl = epoch,self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs):
        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                if not self('begin_epoch'):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self('begin_validate'):
                        self.all_batches()
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally: self('after_fit')

    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_epoch', 'begin_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        #We use the logger function of the `Learner` here, it can be customized to write in a file or in a progress bar
        self.logger(self.train_stats)
        self.logger(self.valid_stats)

def get_learner(nfs, data, lr, layer, loss_func=F.cross_entropy,
                cb_funcs=None, opt_func=sgd_opt, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)