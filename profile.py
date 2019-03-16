# code based on https://github.com/e-lab/pytorch-toolbox/blob/master/profiler/profile.py
import argparse

import torch
import torch.nn as nn

import constants

USE_GPU = constants.USE_GPU

# count operations for 2D convolution
def count_conv2d(m, x, y):
    x = x[0]

    # input and output channels
    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    # size of kernel
    kh, kw = m.kernel_size
    # size of batch
    batch_size = x.size()[0]

    # ops per output element, depends on input channels and kernel size
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops as a function of elements
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

# count operations for 2D convolution transpose, which is the same as a convolution
def count_convtranspose2d(m, x, y):
    x = x[0]

    # input and output channels
    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    # size of kernel
    kh, kw = m.kernel_size
    # size of batch
    batch_size = x.size()[0]

    # ops per output element, depends on input channels and kernel size
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops as a function of elements
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

# count batch normalization operations
def count_bn2d(m, x, y):
    x = x[0]

    # have to normalize by mean, all elements must be subtracted to range and then divided to standardize
    # the deviation
    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    # take in model
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.ConvTranspose2d):
            m.register_forward_hook(count_convtranspose2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    # input of size that we want to profile
    x = torch.zeros(input_size)
    if USE_GPU:
        model = model.cuda()
        x = x.cuda()

    model(x)

    # take operations and parameters from the model by looping through the modules
    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    # return operations and parameters
    return total_ops, total_params

def main(args):
    if USE_GPU:
        model = torch.load(args.model).cuda()
    else:
        model = torch.load(args.model, map_location='cpu')
    input_size = [constants.BatchSize, 1, constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1]] # hardcoded input size
    total_ops, total_params = profile(model, input_size)
    print("#Ops: %f GOps"%(total_ops/1e9))
    print("#Parameters: %f M"%(total_params/1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pytorch model profiler")
    parser.add_argument("model", help="model to profile")

    args = parser.parse_args()
    main(args)
