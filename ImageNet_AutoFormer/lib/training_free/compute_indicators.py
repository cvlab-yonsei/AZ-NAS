from .p_utils import *
from . import indicators

import types
import copy

def no_op(self,x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def find_indicators_arrays(net_orig, trainloader, dataload_info, device, indicator_names=None, loss_fn=F.cross_entropy):
    if indicator_names is None:
        indicator_names = indicators.available_indicators

    dataload, num_imgs_or_batches, num_classes = dataload_info

    net_orig.to(device)
    if not hasattr(net_orig,'get_copy'):
        net_orig.get_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu() 
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    if dataload == 'random':
        inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(trainloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 10
    indicator_values = {}

    while not done:
        try:
            for indicator_name in indicator_names:
                if indicator_name not in indicator_values:
                    if indicator_name == 'NASWOT'  or indicator_name=='te_nas':
                        val = indicators.calc_indicator(indicator_name, net_orig, device)
                    else:
                        val = indicators.calc_indicator(indicator_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
                        indicator_values[indicator_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=True
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return indicator_values

def find_indicators(net_orig,
                  dataloader,
                  dataload_info,
                  device,
                  loss_fn=F.cross_entropy,
                  indicator_names=None,
                  indicators_arr=None):
    

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    if indicators_arr is None:
        indicators_arr = find_indicators_arrays(net_orig, dataloader, dataload_info, device, loss_fn=loss_fn, indicator_names=indicator_names)

    indicators = {}
    for k,v in indicators_arr.items():
        if k == 'NASWOT' or k=='te_nas':
            indicators[k] = v
        else:
            indicators[k] = sum_arr(v)

    return indicators
