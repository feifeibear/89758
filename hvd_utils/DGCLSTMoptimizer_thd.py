from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

from horovod.common import init
from horovod.common import size
from horovod.common import local_size
from horovod.common import rank
from horovod.common import local_rank
from horovod.common import mpi_threads_supported
from horovod.common import check_extension

#check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
#                __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async, _allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import poll, synchronize
import numpy as np
from .pruning import select_top_k_thd, select_top_k_appr, check_sparsity, select_top_k_thdv3, select_top_k_fixthd
import horovod.torch as hvd

import torch


class _DGCOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters=None, use_gpu=True, momentum=0.9, weight_decay=1e-4, use_allgather=True):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        self._parameter_names = {v: k for k, v
                                 in sorted(named_parameters)}
        self._use_gpu = use_gpu
        self._use_nesterov = False
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._debug = False
        self._use_allgather = use_allgather 

        # define U for residue, V for momentum
        if self._use_gpu:
            self._V = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._masks = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._compressed_msg = {k: torch.zeros(0).cuda() for k, v
                                 in sorted(named_parameters)}
        else:
            self._V = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._masks = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._compressed_msg = {k: torch.zeros(0) for k, v
                                 in sorted(named_parameters)}
        self._compressed_msg_size = {k: 0 for k, v
                                 in sorted(named_parameters)}
        self._v_ref = {k: [] for k, v
                                 in sorted(named_parameters)}

        self._handles = {}
        self._grad_accs = []
        self._interval = 20

        self.pruning_time = 0.0
        self.select_time = 0.0
        self.comm_time = 0.0

    def synchronize(self):
        for p in self._handles:
            handle = self._handles[p]
            synchronize(handle)
            p_size = np.prod(p.size())
            begin_time = time.time()

            torch.cuda.synchronize()
            begin_comm_time =  time.time()
            if self._use_allgather and p_size > 1024:
                #fjr decompress
                name = self._parameter_names.get(p)
                msg_size = self._compressed_msg_size[name]
                #print("rank, msg_size is ", hvd.local_rank(), msg_size)
                g_size = p.grad.data.size()
                p_flatten = p.grad.data.view(-1)
                p_flatten.zero_()
                #print("p_flatten size is ,", p_flatten.size())
                #print("compressed msg, ", self._compressed_msg[name], 'rank, ', hvd.local_size())
                #print("hand is ", handle)
                offset = 0
                for node_idx in range(hvd.size()):
                    if self._use_gpu:
                        msg_size = self._compressed_msg[name][offset].type('torch.cuda.LongTensor')
                        offset += 1
                        p_flatten[self._compressed_msg[name][ offset: \
                            offset + msg_size].type('torch.cuda.LongTensor')] += \
                           self._compressed_msg[name][offset + msg_size : \
                           offset + 2*msg_size]
                        offset += msg_size * 2;
                    else:
                        pass

                p.grad.data = p.grad.data.view(g_size)
                if self._debug:
                    print("diff : ", torch.sum(self._v_ref[name] - p.grad.data))

            torch.cuda.synchronize()
            end_time = time.time()
            self.pruning_time += end_time - begin_time
            self.comm_time += time.time() - begin_comm_time

        self._handles.clear()

    def step(self, closure=None):
        # local clipping
        # DGC
        for group in self.param_groups:
            for p in group['params']:
                p.grad.data.add_(torch.mul(p.data, self._weight_decay))
                p.grad.data.div_(hvd.size())

            torch.nn.utils.clip_grad_norm_(group['params'], 0.25 * hvd.size() ** -0.5)
            #torch.nn.utils.clip_grad_norm(group['params'], 0.25)
            #weight_decay = group['weight_decay']
            #momentum = group['momentum']
            torch.cuda.synchronize()
            begin_time =  time.time()

            dampening = 0.0 #gcoup['dampening']
            for p in group['params']:
                assert p not in self._handles
                assert not p.grad.requires_grad
                name = self._parameter_names.get(p)
                p_size = np.prod(p.size())
                if self._use_allgather and p_size > 1024:
                    param_state = self.state[p]
                    self._V[name].add_(p.grad.data)
                    compressed_val = []
                    compressed_idx = []
                    #if p_size < 1000:
                    #self._masks[name], compressed_val, compressed_idx = select_top_k_appr(self._V[name], 0.001, self._masks[name])

                    torch.cuda.synchronize()
                    begin_select_time =  time.time()
                    if 'mid_store' not in param_state:
                        param_state['mid_store'] = 0.0
                    if 'interval' not in param_state:
                        param_state['interval'] = self._interval
                    compressed_val = []
                    compressed_idx = []
                    if param_state['interval'] == self._interval:
                        compressed_val, compressed_idx, it, param_state['mid_store'], sparsity = \
                            select_top_k_thdv3(self._V[name], 0.001)
                        param_state['interval'] = 0
                    else:
                        compressed_val, compressed_idx, sparsity = \
                            select_top_k_fixthd(self._V[name], param_state['mid_store'])
                        param_state['interval'] += 1
                    #masks_size = self._masks[name].size()
                    #self._masks[name].zero_()
                    #self._masks[name] = self._masks[name].view(-1)
                    #self._masks[name][compressed_idx] = 1.0
                    #self._masks[name] = 1.0 - self._masks[name]
                    #self._masks[name] = self._masks[name].view(masks_size)
                    torch.cuda.synchronize()
                    end_select_time =  time.time()
                    self.select_time += end_select_time - begin_select_time
                    if self._debug:
                        self._v_ref[name] = self._V[name] * self._masks[name]
                        allreduce_(self._v_ref[name], average = False)

                    #self._V[name] = self._V[name] * (1 - self._masks[name])
                    #self._U[name] = self._U[name] * (1 - self._masks[name])
                    V_size = self._masks[name].size()
                    self._V[name] = self._V[name].view(-1)
                    self._V[name][compressed_idx] = 0.0
                    self._V[name] = self._V[name].view(V_size)

                    torch.cuda.synchronize()
                    begin_comm_time =  time.time()

                    self._compressed_msg_size[name] = len(compressed_idx)
                    if self._use_gpu:
                        compressed_msg = torch.cat([\
                            torch.tensor([len(compressed_idx)]).type('torch.cuda.FloatTensor'),\
                            compressed_idx.type('torch.cuda.FloatTensor'), \
                            compressed_val])

                    handle = _allgather_async(compressed_msg, self._compressed_msg[name], name=name)
                    self._handles[p] = handle

                    torch.cuda.synchronize()
                    self.comm_time += time.time() - begin_comm_time

                else:
                    handle = allreduce_async_(p.grad.data, average=True, name=name)
                    self._handles[p] = handle

            torch.cuda.synchronize()
            end_time = time.time()
            self.pruning_time += end_time - begin_time

        self.synchronize()
        return super(self.__class__, self).step(closure)


def DGCLSTMDistributedOptimizer(optimizer, named_parameters=None, use_gpu=True, momentum=0.9, weight_decay=1e-4, use_allgather=True):
    """
    An optimizer that wraps another torch.optim.Optimizer, 
    Compress gradients according to their magnitude
    using an allgather to reduce compressed gradient values before applying gradients to model weights.
    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DGCOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,use_gpu, momentum, weight_decay, use_allgather)

