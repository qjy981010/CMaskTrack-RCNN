import torch
from spatial_correlation_sampler import SpatialCorrelationSampler,spatial_correlation_sample 

device = "cuda"
batch_size = 1
channel = 1
H = 10
W = 10
dtype = torch.float32

input1 = torch.randint(1, 4, (batch_size, channel, H, W), dtype=dtype, device=device, requires_grad=True)
input2 = torch.randint_like(input1, 1, 4).requires_grad_(True)

#You can either use the function or the module. Note that the module doesn't contain any parameter tensor.

#function
"""
out = spatial_correlation_sample(input1,
                             input2,
                                 kernel_size=3,
                                 patch_size=1,
                                 stride=2,
                                 padding=0,
                                 dilation=2,
                                 dilation_patch=1)
"""
#module

correlation_sampler = SpatialCorrelationSampler(
    kernel_size=3,
    patch_size=1,
    stride=2,
    padding=0,
    dilation=2,
    dilation_patch=1)
out = correlation_sampler(input1, input2)
import pdb
pdb.set_trace()
