import torch
import torch.nn as nn
from saveNet import *


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
#         real_weight = self.register_buffer('weight', torch.zeros(in_features*out_features))
#         real_bias = self.register_buffer('bias', torch.zeros(in_features))
# 
#         imag_weight = self.register_buffer('weight', torch.zeros(in_features*out_features))
#         imag_bias = self.register_buffer('bias', torch.zeros(in_features))
        
        self.real_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
    def forward(self, input):
        '''    
            assume complex input z = x + iy needs to be convolved by complex filter h = a + ib
            where Output O = z * h, where * is multiplication operator, then O = x*a + i(x*b)+ i(y*a) - y*b
            so we need to calculate each of the 4 convolution operations in the previous equation,
            one simple way to implement this as two real fc layers, one layer for the real weights (a) 
            and the other for imaginary weights (b), this can be done by concatenating both real and imaginary 
            parts of the input and multiply over both of them as follows: 
            c_r = [x; y] * a , and  c_i= [x; y] * b, so that
            O_real = c_r[real] - c_i[real], and O_imag = c_r[imag] - c_i[imag]
        '''
        ndims = input.ndimension()
        input_shape = input.shape
        
        def calc_fc_length(input_shape):
            l = 1
            for i in range(2, len(input_shape)):
                l *= input_shape[i]
            return l
        
        real_input_ln = torch.reshape(input.narrow(ndims-1, 0, 1), (input_shape[0], input_shape[1], calc_fc_length(input_shape[0:-1])))
        imag_input_ln = torch.reshape(input.narrow(ndims-1, 1, 1), (input_shape[0], input_shape[1], calc_fc_length(input_shape[0:-1])))
        
#         if ndims == 4:            
#             real_input_ln = torch.reshape(input[:,:,:,0], 
#                                      (input_shape[0], input_shape[1], input_shape[2]))
# 
#             imag_input_ln = torch.reshape(input[:,:,:,1], 
#                                      (input_shape[0], input_shape[1], input_shape[2]))
#             
#         elif ndims == 5:
#             real_input_ln = torch.reshape(input[:,:,:,:,0], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]))
# 
#             imag_input_ln = torch.reshape(input[:,:,:,:,1], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]))
# 
#         elif ndims == 6:
#             real_input_ln = torch.reshape(input[:,:,:,:,:,0], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]))
# 
#             imag_input_ln = torch.reshape(input[:,:,:,:,:,1], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]))
#         
        
        real_output_ln = self.real_linear(real_input_ln) - self.imag_linear(imag_input_ln)
        imag_output_ln = self.real_linear(imag_input_ln) + self.imag_linear(real_input_ln)
        
        real_output = torch.reshape(real_output_ln, input_shape[0:-1])
        imag_output = torch.reshape(imag_output_ln, input_shape[0:-1])
        
#         if ndims == 4:            
#             real_output = torch.reshape(real_output_ln, input_shape)
# 
#             imag_output = torch.reshape(imag_output_ln, 
#                                      (input_shape[0], input_shape[1], input_shape[2],input_shape[3]))
#             
#         elif ndims == 5:
#             real_input_ln = torch.reshape(input[:,:,:,:,0], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]))
# 
#             imag_input_ln = torch.reshape(input[:,:,:,:,1], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]))
# 
#         elif ndims == 6:
#             real_input_ln = torch.reshape(input[:,:,:,:,:,0], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]*input_shape[5]))
# 
#             imag_input_ln = torch.reshape(input[:,:,:,:,:,1], 
#                                      (input_shape[0], input_shape[1], input_shape[2]*input_shape[3]*input_shape[4]*input_shape[5]))

                
        output = torch.stack((real_output, imag_output),dim=ndims-1)
        
#         output = torch.reshape(output_ln, input_shape)
        
        return output
            