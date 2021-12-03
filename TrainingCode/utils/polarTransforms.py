import torch


def polarToCylindricalConversion(input1, input2=None):
    
    if input2 is None:
        '''input1 is tensor of [B,C,H,W,D,2] contains both magnitude and phase channels
         in the last dims'''
#         if input1.ndimension() == 4:
#             mag_input, phase_input = input1[:,:,:,0], input1[:,:,:,1]
#         elif input1.ndimension() == 5:
#             mag_input, phase_input = input1[:,:,:,:,0], input1[:,:,:,:,1]
#         elif input1.ndimension() == 6:
#             mag_input, phase_input = input1[:,:,:,:,:,0], input1[:,:,:,:,:,1]
        
        ndims = input1.ndimension()
        mag_input = input1.narrow(ndims-1, 0, 1).squeeze(ndims-1)
        phase_input = input1.narrow(ndims-1, 1, 1).squeeze(ndims-1)
          
        real = mag_input * torch.cos(phase_input)
        imag = mag_input * torch.sin(phase_input)
        return torch.stack((real, imag),dim= input1.ndimension()-1)
    else:
        '''input1 is magnitude part and input2 is phase part; both of size [B,C,H,W,D]'''
        real = input1 * torch.cos(input2)
        imag = input1 * torch.sin(input2)
        return real, imag
 
def cylindricalToPolarConversion(input1, input2=None):
    
    if input2 is None: 
        '''input1 is tensor of [B,C,H,W,D,2] contains both real and imaginary channels
         in the last dims'''
#         if input1.ndimension() == 4:
#             real_input, imag_input = input1[:,:,:,0], input1[:,:,:,1]
#         elif input1.ndimension() == 5:
#             real_input, imag_input = input1[:,:,:,:,0], input1[:,:,:,:,1]
#         elif input1.ndimension() == 6:
#             real_input, imag_input = input1[:,:,:,:,:,0], input1[:,:,:,:,:,1]

        ndims = input1.ndimension()
        real_input = input1.narrow(ndims-1, 0, 1).squeeze(ndims-1)
        imag_input = input1.narrow(ndims-1, 1, 1).squeeze(ndims-1)
        
        mag = (real_input**2 + imag_input**2)**(0.5)
        phase = torch.atan2(imag_input, real_input)
        
        phase[phase.ne(phase)] = 0.0 #remove NANs
        return torch.stack((mag, phase), dim = input1.ndimension()-1)        
    else: 
        '''input1 is real part and input2 is imaginary part; both of size [B,C,H,W,D]'''       
        mag = (input1**2 + input2**2)**(0.5)
        phase = torch.atan2(input2, input1)        
        
        phase[phase.ne(phase)] = 0.0 #remove NANs
        return mag, phase


def cylindricalToPolarConversion2(input1, input2=None):
    if input2 is None:
        '''input1 is tensor of [B,C,H,W,D,2] contains both real and imaginary channels
         in the last dims'''

        ndims = input1.ndimension()
        real_input = input1.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        imag_input = input1.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        mag = (real_input ** 2 + imag_input ** 2) ** (0.5)
        phase = torch.atan2(imag_input, real_input)

        phase[phase.ne(phase)] = 0.0  # remove NANs
        return torch.stack((mag, phase), dim=input1.ndimension() - 1)
    else:
        '''input1 is real part and input2 is imaginary part; both of size [B,C,H,W,D]'''
        mag = (input1 ** 2 + input2 ** 2) ** (0.5)
        phase = torch.atan2(input2, input1)

        phase[phase.ne(phase)] = 0.0  # remove NANs
        return mag, phase

# def polarToCylindricalConversion(mag, phase):
# 
#     real = mag * torch.cos(phase)
#     imag = mag * torch.sin(phase)
#     
#     return real, imag
#  
# def cylindricalToPolarConversion(real, imag):
#  
#     mag = (real**2 + imag**2)**(0.5)
#     phase = torch.atan2(imag, real)
#     
#     phase[phase.ne(phase)] = 0.0 #remove NANs
#     
#     return mag, phase
# 
# 
# def polarToCylindricalConversion(input):
#     if input.ndimension() == 4:
#         mag_input, phase_input = input[:,:,:,0], input[:,:,:,1]
#     elif input.ndimension() == 5:
#         mag_input, phase_input = input[:,:,:,:,0], input[:,:,:,:,1]
#     elif input.ndimension() == 6:
#         mag_input, phase_input = input[:,:,:,:,:,0], input[:,:,:,:,:,1]
#      
#     real = mag_input * torch.cos(phase_input)
#     imag = mag_input * torch.sin(phase_input)
#      
#     return torch.stack((real, imag),dim= input.ndimension()-1)
#   
# def cylindricalToPolarConversion(input):
#     if input.ndimension() == 4:
#         real_input, imag_input = input[:,:,:,0], input[:,:,:,1]
#     elif input.ndimension() == 5:
#         real_input, imag_input = input[:,:,:,:,0], input[:,:,:,:,1]
#     elif input.ndimension() == 6:
#         real_input, imag_input = input[:,:,:,:,:,0], input[:,:,:,:,:,1]
#      
#     mag = (real_input**2 + imag_input**2)**(0.5)
#     phase = torch.atan2(imag_input, real_input)
#      
#     phase[phase.ne(phase)] = 0.0 #remove NANs
#     return torch.stack((mag, phase), dim = input.ndimension()-1)