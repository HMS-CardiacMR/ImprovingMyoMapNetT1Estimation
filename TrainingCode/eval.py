#models
from logging import exception
import math
import shutil
import sys

from scipy.io import loadmat, savemat
from tensorboardX import SummaryWriter
from torch import optim
import torch
from torch.autograd import Variable

from myloss import *
import numpy as np
from parameters import Parameters
from saveNet import *
import torch.nn.modules.loss as Loss
import torchvision.utils as vutils
from unet import UNet
from utils import *
from utils.cmplxBatchNorm import magnitude, normalizeComplexBatch_byMagnitudeOnly
from utils.dataset import *


#data manipulation
# import pandas as pd
# import PIL
# from scipy import stats
# import h5py
# import itertools
# from utils.flipTensor import flip
#load files
# import os
# import matplotlib.pyplot as plt
# from os.path import join
# import glob
#data visualization
# import matplotlib.pyplot as plt
#quit after interrupt
# from scipy.io.matlab.mio import savemat
# import argparse
# from graphviz import Digraph
# import re
######################################3
#set seed points
seed_num = 888 

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

####################################
params = Parameters()

# if params.ScoredPatientsOnly:
#     P = loadmat(params.net_save_dir+'lgePatients_score4and3.mat')['lgePatients']
#     pNames = [i[1][0] for i in P]
# 
# if params.Op_Node == 'alpha_V12' or params.Op_Node == 'myPC':
#     if os.path.exists(params.dir1): 
#         for f in os.listdir(params.dir1):
#             if params.ScoredPatientsOnly:
#                 if f in pNames:
#                     params.patients.append(params.dir1 + f)
#             else:
#                 params.patients.append(params.dir1 + f)
#             params.ds_total_num_slices += len(os.listdir(params.dir1+f+'/InputData/Input_realAndImag/'))
#     if os.path.exists(params.dir2):        
#         for f in os.listdir(params.dir2):
#             if params.ScoredPatientsOnly:
#                 if f in pNames:
#                     params.patients.append(params.dir2 + f)
#             else:
#                 params.patients.append(params.dir2 + f)
#             params.ds_total_num_slices += len(os.listdir(params.dir2+f+'/InputData/Input_realAndImag/'))
#         
# elif params.Op_Node == 'O2':
#     for f in os.listdir(params.dir):
#         if params.ScoredPatientsOnly:
#             if f in pNames:
#                 params.patients.append(params.dir + f)
#         else:
#             params.patients.append(params.dir+f)
#         params.ds_total_num_slices += len(os.listdir(params.dir+f+'/InputData/Input_realAndImag/'))
# 
# 
# # params.patients = params.patients[::-1]
# params.patients = params.patients[0:round(params.training_percent*len(params.patients))]    
# params.ds_total_num_slices = round(params.ds_total_num_slices*params.training_percent)






## source activate pytorch0.41
## CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 /data2/helrewaidy/code/RecoNetCmplx-PyTorch4/main.py 


####################################
#
# Create Data Generators
#
####################################

training_DG, validation_DG = getDatasetGenerators()



####################################
#
# Create Model
#
####################################

Fourier_Model = False
if Fourier_Model:
    ''' FC Model for Fourier Transform '''
    from complexnet.cmplxfc import ComplexLinear
    img_size = [64,64]
    fnet = torch.nn.Sequential(ComplexLinear(img_size[0]*img_size[1], img_size[0]*img_size[1]),
                         torch.nn.ReLU(),
                         ComplexLinear(img_size[0]*img_size[1], img_size[0]*img_size[1])
                         ).to(params.device)
else:
    net = UNet(params.n_channels, 1)
    
    if params.multi_GPU:
        net = torch.nn.DataParallel(net,device_ids=params.device_ids).cuda()
    else:
        net.to(params.device)
    
    

####################################
#
# PyTorch Hooks
#
####################################
tbv_itt = 0;
def visualize_featureMaps_hook(module, input, output):
#     outputs.append(output)
    ndims = output.ndimension()
    tbv_itt = 0
    if ndims > 1:
        tbVisualizeImage(output.detach(), module._get_name(), tbv_itt, visualize_magnitudes=True)
        tbVisualizeImage(output.detach(), module._get_name(), tbv_itt, visualize_magnitudes=False)
    else:
        for i in range(len(output)): 
            tbVisualizeImage(output[i].detach(), module._get_name(), tbv_itt, visualize_magnitudes=True)
            tbVisualizeImage(output[i].detach(), module._get_name(), tbv_itt, visualize_magnitudes=False)
    

def nan_checks_hook(model, gradInput, gradOutput):    
    def check_grad(module, grad_input, grad_output):
#         print(module) # you can add this to see that the hook is called
#         print('hook !!!')
        if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
    print('gradInput',gradInput,'/n/n/n')
    print('gradOutput',gradOutput,'/n/n/n')
    model.apply(lambda module: module.register_backward_hook(check_grad))


### Register hooks
# net.register_backward_hook(nan_checks_hook)
# net.register_forward_hook(nan_checks_hook)

# net.inc.conv.conv[0].conv_func_calculation.real_conv.register_backward_hook(nan_checks_hook)
# net.inc.conv.conv[0].conv_func_calculation.imag_conv.register_backward_hook(nan_checks_hook)

##>>>>>> Access inside sequential layer <<<<<<<<<<<<
# net.inc.conv.conv[0].conv_func_calculation.conv.weight


# net.down1.mpconv.register_forward_hook(visualize_featureMaps_hook)
# net.down2.mpconv.register_forward_hook(visualize_featureMaps_hook)
# net.bottleneck.mpconv.register_forward_hook(visualize_featureMaps_hook)
#  
# net.up2.register_forward_hook(visualize_featureMaps_hook)
# net.up2.up.register_forward_hook(visualize_featureMaps_hook)
# net.up2.up.conv_func_calculation.register_forward_hook(visualize_featureMaps_hook)
#  
# net.up3.register_forward_hook(visualize_featureMaps_hook)
# net.up3.up.register_forward_hook(visualize_featureMaps_hook)
# net.up3.up.conv_func_calculation.register_forward_hook(visualize_featureMaps_hook)
#  
# net.up4.register_forward_hook(visualize_featureMaps_hook)
# net.up4.up.register_forward_hook(visualize_featureMaps_hook)
# net.up4.up.conv_func_calculation.register_forward_hook(visualize_featureMaps_hook)

####################################
#
# initializations 
#
####################################

if not os.path.exists(params.model_save_dir):
    os.makedirs(params.model_save_dir)

if not os.path.exists(params.tensorboard_dir):
    os.makedirs(params.tensorboard_dir)

writer = SummaryWriter(params.tensorboard_dir)

####################################################################
## Fix bug: "Can't get attribute '_rebuild_tensor_v2' on <module 'torch._utils'"
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
#################################################################

def train(net):
    
    ###########################################
    #
    # INITIALIZATIONS
    #
    ############################################
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    LOSS = list()
    vld_MSE_LOSS = list()
    vld_SSIM_LOSS = list()
    vld_mse = 0
    vld_ssim = 0
        
    diceCriterion = DiceLoss()
    ssimCriterion = SSIM()
    mseCriterion = Loss.MSELoss()
    kscCriterion = KspaceConsistency()
    tvCriterion = TotalVariations()
    
#     lossCriterions = [mseCriterion, ssimCriterion, kscCriterion, tvCriterion];
    lossCriterions = [mseCriterion];
    
    lambda_mse = Variable(torch.FloatTensor([0.9]),requires_grad=True).to(params.device)
    lambda_ksc = Variable(torch.FloatTensor([4.5e6]),requires_grad=True).to(params.device)
    lambda_tv = Variable(torch.FloatTensor([4.2e5]),requires_grad=True).to(params.device)
    lambda_ssim = Variable(torch.FloatTensor([0.2]),requires_grad=True).to(params.device)
    
    MSELoss = list()
    KSCLoss = list()
    TVLoss = list()
    SSIMLoss = list()
    
    vi = 0
    i = 0
    bt = 0
    
    
    ###########################################
    #
    # LOAD LATEST (or SPECIFIC) MODEL
    #
    ############################################
    models = os.listdir(params.model_save_dir);
    epoch_nums = [int(epo[11:-4]) for epo in models[:]]
    epoch_nums.sort()
    s_epoch = 0;
#     if len(models) > 0:
#         if s_epoch==0:
#             s_epoch = max([int(epo[11:-4]) for epo in models[:]])
#         print("Loading model ...")
# #         net.load_state_dict(torch.load(model_save_dir+models[1][0:11]+str(s_epoch)+'.pth'))
#         net.load_state_dict(torch.load(params.model_save_dir+models[0][0:11]+str(s_epoch)+'.pth')['state_dict'])
#         optimizer.load_state_dict(torch.load(params.model_save_dir+models[0][0:11]+str(s_epoch)+'.pth')['optimizer'])
#         LOSS = torch.load(params.model_save_dir+models[0][0:11]+str(s_epoch)+'.pth')['loss']
#         s_epoch = s_epoch - 1
#         print("Model loaded !")
    Evaluate_Last_Epoch = False
    if Evaluate_Last_Epoch:
        epoch_nums = [epoch_nums[-1],]    
    
    for epoch in epoch_nums:

        net.load_state_dict(torch.load(params.model_save_dir+models[0][0:11]+str(epoch)+'.pth')['state_dict'])
#         optimizer.load_state_dict(torch.load(params.model_save_dir+models[0][0:11]+str(epoch)+'.pth')['optimizer'])
#         LOSS = torch.load(params.model_save_dir+models[0][0:11]+str(epoch)+'.pth')['loss']    
        
        print('epoch {}/{}...'.format(epoch+1, params.epochs))
        
        #####################################
        #
        # Validation
        #
        #####################################
        vl = 0
        vitt = 0
        TAG = 'Validation'
        with torch.no_grad():
            for local_batch, local_labels in validation_DG:
                
                X = Variable(torch.FloatTensor(local_batch.float())).to(params.device)
                X = normalizeComplexBatch_byMagnitudeOnly(X, False)
                 
                y = Variable(torch.FloatTensor(local_labels.float())).to(params.device)
                y = normalizeComplexBatch_byMagnitudeOnly(y, True)                      
                       
                if params.tbVisualize:
                    tbVisualizeImage(X, TAG+'/'+str(epoch)+'/'+str(itt)+'/1-Input', itt)                                                
                    tbVisualizeImage(y[:,:,:,:,0], TAG+'/'+str(epoch)+'/'+str(itt)+'/3-Output',itt ,True)
                    
                y_pred = net(X)
                
                if params.tbVisualize:
                    tbVisualizeImage(y_pred.detach(), TAG+'/'+str(epoch)+'/'+str(itt)+'/2-Predictions',itt , True) 
                    tbVisualizeImage(y_pred.detach(), TAG+'/'+str(epoch)+'/'+str(itt)+'/2-Predictions',itt , False) 
                        
                mseloss = mseCriterion(magnitude(y_pred).squeeze(1), y[:,:,:,:,0].squeeze(1))                   
                if True or params.tbVisualize:
                    writer.add_scalar(TAG+'/'+'MSE_Loss', mseloss,vi)
                                 
                ssimloss = ssimCriterion(magnitude(y_pred), y[:,:,:,:,0])
                if True or params.tbVisualize:
                    writer.add_scalar(TAG+'/'+'SSIM_Loss', ssimloss,vi)
    
                vld_MSE_LOSS.append(mseloss.cpu().data.numpy())
                vld_SSIM_LOSS.append(ssimloss.cpu().data.numpy())
    
                vld_mse += mseloss.data[0]
                vld_ssim += ssimloss.data[0]
                
                vi +=1
                vitt += 1
                print( 'Epoch: {0} - {1:.3f}%'.format(epoch+1,100*(vitt*params.batch_size)/len(validation_DG.dataset.input_IDs)) +
                        ' \tIter: ' + str(vi) + '\tSME: {0:.6f}'.format(mseloss.data[0])+ '\tSSIM: {0:.6f}'.format(ssimloss.data[0]))
                
            if True or params.tbVisualize:
                writer.add_scalar(TAG+'/'+'avg_SME', vld_mse/len(validation_DG.dataset.input_IDs) ,epoch)
                writer.add_scalar(TAG+'/'+'avg_SSIM', vld_ssim/len(validation_DG.dataset.input_IDs) ,epoch)
            
    writer.close()


def tbVisualizeImage(x, tag, itt, visualize_magnitudes=True):
    def validate_dims(T):
        if T.shape[1] > 1:
            T = T.reshape((T.shape[0]*T.shape[1],T.shape[2], T.shape[3])).unsqueeze(1)
        return T
    
    if x.ndimension() == 5 and not visualize_magnitudes:
        x_r = validate_dims(x[:,:,:,:,0].squeeze(-1))
        x_i = validate_dims(x[:,:,:,:,0].squeeze(-1))
        tb_x={'real':torch.cat([x_r, x_r, x_r],dim=1)/6,
              'imag':torch.cat([x_i, x_i, x_i],dim=1)/6}        
    else:
        if x.ndimension() < 4:
            try:
                writer.add_histogram(tag, x, itt)
            except:
                print('NANs in --> ' + tag)
                pass
            return
        if visualize_magnitudes and x.ndimension() == 5:
            x = magnitude(x)
        x = validate_dims(x)
        tb_x = {'magnitude':torch.cat([x, x, x],dim=1)/6}
    
    for name, T in tb_x.items():
        writer.add_image(tag+'/'+name, vutils.make_grid(T, nrow=4, normalize=True), itt)
        try:
            writer.add_histogram(tag+'/'+name, T, itt)
        except:
            print('NANs in --> ' + tag)
            pass

    
def normalizeMPImages(img):
    img[0:256,:,:] = (img[0:256,:,:] - np.mean(img[0:256,:,:], axis=(0,1)))/np.std(img[0:256,:,:], axis=(0,1))
    img[256:,:,:] = (img[256:,:,:] - np.mean(img[256:,:,:], axis=(0,1)))/np.std(img[256:,:,:], axis=(0,1))
    return img

def normalizeBatch(p):
    ''' normalize each slice alone'''
    return (p - np.mean(p, axis=(0,1)))/np.std(p, axis=(0,1))

def normalizeComplexBatch(p):
    ''' normalize each slice alone'''
    return (p - np.mean(p, axis=(0,1)))/complexSTD(p)

def complexSTD(x):
    ''' 
    Standard deviation of real and imaginary channels
    STD = sqrt( E{(x-mu)(x-mu)*} ), where * is the complex conjugate, 
        
    - Source: https://en.wikipedia.org/wiki/Variance#Generalizations
    '''
    mu = np.mean(x,axis=(0,1))
    xm = np.sum((x-mu)**2, axis=2); #(a+ib)(a-ib)* = a^2 + b^2
    return np.sqrt( np.mean( xm , axis=(0,1)))
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def polarToRectangularConversion(magnitude, phase):
    real = magnitude*np.cos(phase)
    imaginary = magnitude*np.sin(phase)
    return np.dstack((real, imaginary))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params.args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def get_slices_by_index(data, indx):
    ''' 
    This function returns sub-set of 'data' based on 'indx'
    
    '''
    indx.sort()
    outdata = []
    slices_num = [s.shape[2] for s in data][::-1]
    sl_num = 0
    i=0
    for D in data: 
#         s_idnx = [j for j in indx if j<=slices_num[i]+sl_num and j>sl_num]
        s_idnx = []
        for j in indx:
            if j<=slices_num[i]+sl_num and j>sl_num:
                s_idnx.append(j)
                
        sl_num += slices_num[i] 
        i += 1
        outdata.append([D[:,:,f,:,:] for f in s_idnx])
        
try:
#     net.load_state_dict(torch.load('MODEL.pth'))
    train(net)

except KeyboardInterrupt:
    print('Interrupted')
    torch.save(net.state_dict(), 'MODEL_INTERRUPTED.pth')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)