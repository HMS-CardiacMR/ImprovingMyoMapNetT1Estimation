# models
from logging import exception
import math
import shutil
import sys
#import time
import pyCompare
import pingouin as pg
import scipy
from scipy.io import loadmat, savemat
# from tensorboardX import SummaryWriter
from torch import optim
import matplotlib.pyplot as plt
#import torch
#from torch.autograd import Variable

from myloss import *
#import numpy as np
#from parameters import Parameters
#from saveNet import *
import torch.nn.modules.loss as Loss
import torchvision.utils as vutils
from FCMyoMapNet import UNet
from utils import *
from utils.cmplxBatchNorm import magnitude, normalizeComplexBatch_byMagnitudeOnly, log_mag, exp_mag
#from utils.dataset import *
from utils.fftutils import *
from utils.data_vis import *
#from utils.data_vis import exportPNG
#from utils.gaussian_fit import kspaceImg_gauss_fit2
#from utils.gridkspace import get_grid_neighbors_mp
import traceback
from myloss import weighted_mse
#from utils.dataset_t1mapping import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
#----------------------------------------------------------#
#---------Initialization and Golbal setting----------------#
#----------LOAD LATEST (or SPECIFIC) MODEL-----------------#
#----------------------------------------------------------#
#scaling factor for input inverion time and output T1,
TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling
T1sigNum = 4
T1sigAndTi = T1sigNum*2;
# set seed points
seed_num = 888

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
params = Parameters()   #load parameter

T1sigNum = params.inputLen
# T1sigAndTi = T1sigNum*2+1
T1sigAndTi = T1sigNum*2
# Create Model
#MyoMapNet = UNet(10, 1)         #5HBsPre5, 5HBsPre5Post5, 5HBsPre5Post41
#MyoMapNet = UNet(8, 1)         #4HBsPost4, 4HBsPre4Post4
MyoMapNet = UNet(T1sigAndTi, 1)
# MyoMapNet = UNet(40, 1)  #for 4HBsPre33
LOSS = list()  #training loss
myoAvgLossAllEpochs = list()  #validation loss
bpAvgLossAllEpochs = list()     #validatino loss
itNum = 0
# Print the total parameters

def multiply_elems(x):
    m = 1
    for e in x:
        m *= e
    return m
num_params = 0

for parameters in MyoMapNet.parameters():
    num_params += multiply_elems(parameters.shape)
print('Total number of parameters: {0}'.format(num_params))

#loading trained model
if params.multi_GPU:
    MyoMapNet = torch.nn.DataParallel(MyoMapNet, device_ids=params.device_ids[:-1]).cuda()
else:
    MyoMapNet.to(params.device)

if not os.path.exists(params.model_save_dir):
    os.makedirs(params.model_save_dir)

if not os.path.exists(params.tensorboard_dir):
    os.makedirs(params.tensorboard_dir)

if not os.path.exists(params.validation_dir):
    os.makedirs(params.validation_dir)
# writer = SummaryWriter(params.tensorboard_dir)


##  0: don't load any model; start from model #1
##  num: load model #num

optimizer = optim.SGD(MyoMapNet.parameters(), lr=params.args.lr, momentum=0.8)

w_mae = weighted_mae()
w_mse = weighted_mse()

models = os.listdir(params.model_save_dir)
models = [m for m in models if m.endswith('.pth')]
s_epoch = 945 ## -1: load latest model or start from 1 if there is no saved models. Currently, 945, 360 for model without phantom data
print(len(models))

if s_epoch == -1:
    if len(models) == 0:
        s_epoch = 1
    else:
        #loading the latest model
        try:
            s_epoch = max([int(epo[11:-4]) for epo in models[:]])
            #s_epoch = 2199
            print('loading model at epoch ' + str(s_epoch))
            model = torch.load(params.model_save_dir + models[0][0:11] + str(s_epoch) + '.pth')
            MyoMapNet.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimizer'])

            itNum = model['iteration']
            lossTmp = model['loss']
            # LOSS = lossTmp.tolist()
            print(LOSS)
            itNum = model['iteration']
            lossTmp = loadmat('{0}mse_R{1}_Trial{2}'.format(params.tensorboard_dir, str(params.Rate), params.trialNum))['mse']
            LOSS = lossTmp.tolist()

            myoAvgLossTmp = loadmat(
                '{0}avgLoss_Myo_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
                'myoAvgLossAllEpochs']  # load training loss
            myoAvgLossAllEpochs = myoAvgLossTmp.tolist()  # validation loss

            bpAvgLossTmp = loadmat(
                '{0}avgLoss_BP_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
                'bpAvgLossAllEpochs']  # load training loss
            bpAvgLossAllEpochs = bpAvgLossTmp.tolist()  # validatino loss

        except:
            print('Model {0} does not exist!'.format(s_epoch))
elif s_epoch == 0:
    s_epoch = 1   #creat a model
else:
    try:
        #load specific model
        # s_epoch = max([int(epo[11:-4]) for epo in models[:]])
        print('loading model at epoch ' + str(s_epoch))
        model = torch.load(params.model_save_dir + models[0][0:11] + str(s_epoch) + '.pth')
        # LOSS = model['loss']
        # print(LOSS)
        itNum = model['iteration']

        MyoMapNet.load_state_dict(model['state_dict'])
        optimizer.load_state_dict(model['optimizer'])
        lossTmp = loadmat('{0}mse_R{1}_Trial{2}'.format(params.tensorboard_dir, str(params.Rate), params.trialNum))['mse']    #load training loss
        LOSS = lossTmp.tolist()

        myoAvgLossTmp = loadmat(
            '{0}avgLoss_Myo_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
            'myoAvgLossAllEpochs']  # load training loss
        myoAvgLossAllEpochs = myoAvgLossTmp.tolist()  # validation loss

        bpAvgLossTmp = loadmat(
            '{0}avgLoss_BP_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
            'bpAvgLossAllEpochs']  # load training loss
        bpAvgLossAllEpochs = bpAvgLossTmp.tolist()  # validatino loss
    except:
        print('Model {0} does not exist!'.format(s_epoch))

## copy the code with the model saving directory
os.system("cp -r {0} {1}".format(os.getcwd(), params.model_save_dir))
print('Model copied!')

def convertndarrytochar(inputarr):
        return 0


#---------------------------------------------------------------------------#
#--------------------------------Loading Data-------------------------------#
#---------------------------------------------------------------------------#
#loading dataset

if params.Training_Only:
    tr_N = 0
    # loading training dateset (simulation)
    if params.training_with_Sim:
        print('Start loading training dataset from numerical simulation')
        #read mat file
        #data = loadmat(params.preMyoSimFile)['data']
        matFile = h5py.File(params.TrainSimFile)
        data = matFile['data']
        T1arr = np.array(data['T1'])       #from signal without noise

        ANoisy = np.array(data['A_noisy'])
        BNoisy =  np.array(data['B_noisy'])
        T1Noisy =  np.array(data['T1_noisy'])

        T1wNoisy =  np.array(data['T1w_noisy'])
        Tiarr =  np.array(data['Ti'])

        #80% samples are used in training
        totalSamples = T1arr.shape[0]
        nSampleForTraining = np.fix(totalSamples).astype(int)
        # The simulated signals are stored into a Matrix. The size of matrix is same with that of in-vivo image

        ## Image size is
        sx, sy = 182, 208
        # sx, sy = 50, 50
        tr_N = np.fix(nSampleForTraining/(sx*sy)).astype(int)

        tr_t1w_TI = np.zeros((tr_N, 1, T1sigAndTi, sx, sy))     # include signals and corrsponding TI times
        tr_T1 = np.zeros((tr_N, 1, sx, sy))


        tr_A = np.zeros((tr_N, 1, sx, sy))
        tr_B = np.zeros((tr_N, 1, sx, sy))
        tr_T1_noNoisy = np.zeros((tr_N, 1, sx, sy))


        tr_mask = np.ones((tr_N, 1, sx, sy))
        tr_LVmask = np.zeros((tr_N, 1, sx, sy))
        tr_ROImask = np.zeros((tr_N, 1, sx, sy))
        tr_sliceID = list()

        for ix in range (0,tr_N):

            ixRange = range(ix*sx*sy,(ix+1)*sx*sy)
            t1wSigsTmp = T1wNoisy[0:T1sigNum,ixRange]
            # tiTimesTmp = Tiarr[0:T1sigNum+1,ixRange]*TimeScalingFactor   #for all eight images
            tiTimesTmp = Tiarr[0:T1sigNum, ixRange] * TimeScalingFactor
            t1NoisyTmp = T1Noisy[ixRange,0]*TimeScalingFactor
            #rest are not used
            t1Tmp = T1arr[ixRange,0]*TimeScalingFactor

            ATmp = ANoisy[ixRange,0]
            BTmp = BNoisy[ixRange,0]


            t1wCTi = np.concatenate((t1wSigsTmp.transpose(),tiTimesTmp.transpose()),axis=1).reshape(sx,sy,T1sigAndTi)
            t1wCTi = np.transpose(t1wCTi,(2,0,1))

            #input is magnitude signals
            tr_t1w_TI[ix,0,:,:,:] = np.abs(t1wCTi)
            tr_T1[ix,0,:,:] = t1Tmp.reshape(sx,sy)

            tr_A[ix,0,:,:] = ATmp.reshape(sx,sy)
            tr_B[ix, 0, :, :] = BTmp.reshape(sx, sy)


        #90% for training and 10% for validation
        val_N = int(np.fix(tr_N*0.1))

        sim_Val_N = val_N

        trn_N = tr_N -val_N

        val_T1 = tr_T1[trn_N:tr_N,:,:,:]
        val_Myomask = tr_mask[trn_N:tr_N,:,:,:]
        val_t1w_TI = tr_t1w_TI[trn_N:tr_N,:,:,:,:]

        val_sliceID = list()
        for ix in range(0, val_N):
            tmpstr = ''+chr(np.mod(ix,65))
            # for ij in range(0, 10):
            #     tmpstr = 'Val' + chr(val_sliceID_input[ij, ix])
            val_sliceID.append(tmpstr)
        #divide data into training and validatino

        tr_mask = tr_mask[0:trn_N-1,: ,:,:]
        tr_T1 = tr_T1[0:trn_N-1,: ,:,:]
        tr_t1w_TI = tr_t1w_TI[0:trn_N-1,:,:,:,:]
        tr_N = tr_mask.shape[0]
        #


    if params.training_with_Phantom:

        print('Start loading Phantom training dataset')

        if T1sigNum == 4:
            t1wtiIdx = [0, 1, 2, 3, 6,7,8,9]
            # t1wtiIdx = [0, 1, 2, 3, 8, 9, 10, 11]
            # t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8,
            #             10, 11, 12, 13, 15, 16, 17, 18,
            #             20, 21, 22, 23, 25, 26, 27, 28,
            #             30, 31, 32, 33, 35, 36, 37, 38,
            #             40, 41, 42, 43, 45, 46, 47, 48]
        else:
            t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # read pre-contrast mat file
        data = h5py.File(params.TrainPhantomFile)
        tr_T1_tmp = data['allT1Maps'][:]  # T1
        torch_data = torch.from_numpy(tr_T1_tmp)
        torch_data_per = torch_data.permute((2, 1, 0))
        tr_T1_3 = torch_data_per.numpy()
        tr_T1_arr = np.zeros((tr_T1_3.shape[0], 1, tr_T1_3.shape[1], tr_T1_3.shape[2]))
        # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, tr_T1_3.shape[0]):
            tr_T1_arr[ix, 0, :, :] = tr_T1_3[ix, :, :]
        if tr_N >0:
            tr_T1 = np.concatenate([tr_T1, tr_T1_arr], axis = 0)
        else:
            tr_T1 = tr_T1_arr

        tr_mask_tmp = data['allMask'][:]  # T1
        torch_mask = torch.from_numpy(tr_mask_tmp)
        torch_mask_per = torch_mask.permute((2, 1, 0))
        torch_mask_3 = torch_mask_per.numpy()
        tr_mask_arr = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
        # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, torch_mask_3.shape[0]):
            tr_mask_arr[ix, 0, :, :] = torch_mask_3[ix, :, :]

        if tr_N>0:
            tr_mask = np.concatenate([tr_mask,tr_mask_arr], axis = 0)
        else:
            tr_mask = tr_mask_arr

        tr_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis
        # torch_data = torch.from_numpy(tr_t1w_TI_tmp[:,:,[0,1,2,3,5,6,7,8],:,:])
        torch_data = torch.from_numpy(tr_t1w_TI_tmp[:, :, t1wtiIdx, :, :])
        torch_data_per = torch_data.permute((4, 3, 2, 1, 0))
        torch_data_per_numpy = torch_data_per.numpy()
        if tr_N > 0:
            tr_t1w_TI = np.concatenate([tr_t1w_TI,torch_data_per_numpy],axis=0)
        else:
            tr_t1w_TI = torch_data_per_numpy
        # tr_B = loadmat(params.preInvivoFile)['allBsigs']  #B
        # tr_A = loadmat(params.preInvivoFile)['allAsigs']  #A
        # tr_T1star = loadmat(params.preInvivoFile)['allT1stMaps']  #T1*
        # tr_T1_5 = tr_T1
        tr_N = tr_T1.shape[0]

        print('End loading Phantom training dataset')
    if params.validation_with_Phantom:
        print('Start loading Phantom validation dataset')
        if T1sigNum == 4:
            t1wtiIdx = [0, 1, 2, 3, 6,7,8,9]
            # t1wtiIdx = [0, 1, 2, 3, 8, 9, 10, 11]
            # t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8,
            #             10, 11, 12, 13, 15, 16, 17, 18,
            #             20, 21, 22, 23, 25, 26, 27, 28,
            #             30, 31, 32, 33, 35, 36, 37, 38,
            #             40, 41, 42, 43, 45, 46, 47, 48]
        else:
            t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # read pre-contrast mat file
        data = loadmat(params.ValidationPhantomFile)
        val_T1_tmp = data['allT1Maps'][:]  # T1
        torch_data = torch.from_numpy(val_T1_tmp)
        torch_data_per = torch_data.permute((2, 1, 0))
        val_T1_3 = torch_data.numpy()
        val_T1_arr = np.zeros((val_T1_3.shape[0], 1, val_T1_3.shape[1], val_T1_3.shape[2]))
        # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, val_T1_3.shape[0]):
            val_T1_arr[ix, 0, :, :] =val_T1_3[ix, :, :]

        if val_N >0:
            val_T1 = np.concatenate([val_T1, val_T1_arr], axis = 0)
        else:
            val_T1 = val_T1_arr;

        #load Mask

        val_mask_tmp = data['allMask'][:]  # T1
        torch_mask = torch.from_numpy(val_mask_tmp)
        torch_mask_per = torch_mask.permute((2, 1, 0))
        torch_mask_3 = torch_mask.numpy()
        val_Myomask_arr = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
        #tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, torch_mask_3.shape[0]):
            val_Myomask_arr[ix, 0, :, :] = torch_mask_3[ix, :, :]

        if val_N >0:
            val_Myomask = np.concatenate([val_Myomask,val_Myomask_arr],axis = 0)
        else:
            val_Myomask = val_Myomask_arr

        #load BP Mask
        # val_mask_tmp = data['bpMask'][:]  # T1
        # torch_mask = torch.from_numpy(val_mask_tmp)
        # torch_mask_per = torch_mask.permute((2, 1, 0))
        # torch_mask_3 = torch_mask_per.numpy()
        # val_BPmask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        # for ix in range(0, torch_mask_3.shape[0]):
        #     val_BPmask[ix, 0, :, :] = torch_mask_3[ix, :, :]

        val_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis

        #torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, [0,1,2,3,5,6,7,8], :, :])
        torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, t1wtiIdx, :, :])
        torch_data_per = torch_data.permute((4, 3, 2, 1, 0))
        if val_N>0:
            val_t1w_TI = np.concatenate([val_t1w_TI,torch_data.numpy() ],axis = 0)
        else:
            val_t1w_TI = torch_data.numpy()

        val_N = val_t1w_TI.shape[0]
        ph_val_N = val_N - sim_Val_N
        try:
            subjectsLists = data['subjectsLists']
            subjectsLists = subjectsLists.tolist()

        except Exception as e:
            print(e)

        print('end loading phantom validation dataset')
    elif params.training_with_Invivo:

    # loading in-vivo training data
        print('Start loading training dataset')


        if T1sigNum == 4:
            t1wtiIdx = [0,  1, 2,   3,  5,  6,  7,  8]
            # t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8,
            #             10, 11, 12, 13, 15, 16, 17, 18,
            #             20, 21, 22, 23, 25, 26, 27, 28,
            #             30, 31, 32, 33, 35, 36, 37, 38,
            #             40, 41, 42, 43, 45, 46, 47, 48]
        else:
            t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8,9]

        # read pre-contrast mat file
        data = h5py.File(params.TrainPrePostInvivo_5HBs)
        tr_T1_tmp = data['allT1Maps'][:]  #T1
        torch_data = torch.from_numpy(tr_T1_tmp)
        torch_data_per = torch_data.permute((2,1,0))
        tr_T1_3 = torch_data_per.numpy()
        tr_T1 = np.zeros((tr_T1_3.shape[0],1,tr_T1_3.shape[1],tr_T1_3.shape[2]))
        #tr_mask = np.ones(tr_T1.shape)
        for ix in range(0,tr_T1_3.shape[0]):
            tr_T1[ix,0,:,:] = tr_T1_3[ix,:,:]

        tr_mask_tmp = data['allMask'][:]  #T1
        torch_mask = torch.from_numpy(tr_mask_tmp)
        torch_mask_per = torch_mask.permute((2,1,0))
        torch_mask_3 = torch_mask_per.numpy()
        tr_mask = np.zeros((torch_mask_3.shape[0],1,torch_mask_3.shape[1],torch_mask_3.shape[2]))
        #tr_mask = np.ones(tr_T1.shape)
        for ix in range(0,torch_mask_3.shape[0]):
            tr_mask[ix,0,:,:] = torch_mask_3[ix,:,:]

        tr_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis
        #torch_data = torch.from_numpy(tr_t1w_TI_tmp[:,:,[0,1,2,3,5,6,7,8],:,:])
        torch_data = torch.from_numpy(tr_t1w_TI_tmp[:, :, :, :, :])
        torch_data_per = torch_data.permute((4,3,2,1,0))
        tr_t1w_TI = torch_data_per.numpy()
        # tr_B = loadmat(params.preInvivoFile)['allBsigs']  #B
        # tr_A = loadmat(params.preInvivoFile)['allAsigs']  #A
        # tr_T1star = loadmat(params.preInvivoFile)['allT1stMaps']  #T1*
        tr_T1_5 = tr_T1
        tr_N = tr_T1.shape[0]


        print('End loading pre-contrast training dataset')


    # plt.imshow(tr_T1[1800,0,:,:], cmap='plasma', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    # plt.show()
    #
    # plt.imshow(tr_t1w_TI[100,0,5,:,:], cmap='plasma', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    # plt.show()


    ##loading validation data


    # print('Start loading in-vivo validation dataset')
    # # read pre-contrast mat file
    # data = h5py.File(params.ValidationPrePostInvivo_5HBs)
    # val_T1_tmp = data['allT1Maps'][:]  # T1
    # torch_data = torch.from_numpy(val_T1_tmp)
    # torch_data_per = torch_data.permute((2, 1, 0))
    # val_T1_3 = torch_data_per.numpy()
    # val_T1 = np.zeros((val_T1_3.shape[0], 1, val_T1_3.shape[1], val_T1_3.shape[2]))
    # # tr_mask = np.ones(tr_T1.shape)
    # for ix in range(0, val_T1_3.shape[0]):
    #     val_T1[ix, 0, :, :] =val_T1_3[ix, :, :]
    #
    # #load LV and BP Mask
    # val_mask_tmp = data['myobpMask'][:]  # T1
    # torch_mask = torch.from_numpy(val_mask_tmp)
    # torch_mask_per = torch_mask.permute((2, 1, 0))
    # torch_mask_3 = torch_mask_per.numpy()
    # val_LVBpmask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    # # tr_mask = np.ones(tr_T1.shape)
    # for ix in range(0, torch_mask_3.shape[0]):
    #    val_LVBpmask[ix, 0, :, :] = torch_mask_3[ix, :, :]
    #
    # #load Myo Mask
    #
    # val_mask_tmp = data['myoMask'][:]  # T1
    # torch_mask = torch.from_numpy(val_mask_tmp)
    # torch_mask_per = torch_mask.permute((2, 1, 0))
    # torch_mask_3 = torch_mask_per.numpy()
    # val_Myomask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    # #tr_mask = np.ones(tr_T1.shape)
    # for ix in range(0, torch_mask_3.shape[0]):
    #     val_Myomask[ix, 0, :, :] = torch_mask_3[ix, :, :]
    # #load BP Mask
    # val_mask_tmp = data['bpMask'][:]  # T1
    # torch_mask = torch.from_numpy(val_mask_tmp)
    # torch_mask_per = torch_mask.permute((2, 1, 0))
    # torch_mask_3 = torch_mask_per.numpy()
    # val_BPmask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    # # tr_mask = np.ones(tr_T1.shape)
    # for ix in range(0, torch_mask_3.shape[0]):
    #     val_BPmask[ix, 0, :, :] = torch_mask_3[ix, :, :]
    #
    # val_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis
    #
    # #torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, [0,1,2,3,5,6,7,8], :, :])
    # torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, :, :, :])
    # torch_data_per = torch_data.permute((4, 3, 2, 1, 0))
    # val_t1w_TI = torch_data_per.numpy()
    #
    # val_sliceID_input = data['subjectsLists'][:]
    # val_sliceID = list()
    # for ix in range(0, val_sliceID_input.shape[1]):
    #     tmpstr = ''
    #     for ij in range(0, val_sliceID_input.shape[0]):
    #         tmpstr =tmpstr+ chr(val_sliceID_input[ij,ix])
    #     val_sliceID.append(tmpstr)
    #
    # # torch_slice = torch.from_numpy(tst_sliceID)
    # # tst_sliceID = tst_sliceID.tostring().decode("ascii")
    # # tst_sliceID
    # # tr_B = loadmat(params.preInvivoFile)['allBsigs']  #B
    # # tr_A = loadmat(params.preInvivoFile)['allAsigs']  #A
    # # tr_T1star = loadmat(params.preInvivoFile)['allT1stMaps']  #T1*
    # # tr_T1_5 = tr_T1
    # # tr_N = tr_T1.shape[0]
    # # plt.imshow(tst_T1[0,0,:,:], cmap='plasma', vmin=0, vmax=3)
    # # plt.colorbar()
    # # plt.xticks(())
    # # plt.yticks(())
    # # plt.title("T1 maps")
    # # plt.show()
    # print('End loading validation dataset')


else:
    ##Loading Testing dataset

    print('Start loading testing dataset')
    t1wtiIdx = np.arange(0,T1sigAndTi)

    t1wtiIdx = [0, 1, 2, 3, 8, 9, 10,11]  #for simulatino testing
    t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7] #for phantom testing and in-vivo testing
    # if T1sigNum == 4:
    #     t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8]
    #     # t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8,
    #     #             10, 11, 12, 13, 15, 16, 17, 18,
    #     #             20, 21, 22, 23, 25, 26, 27, 28,
    #     #             30, 31, 32, 33, 35, 36, 37, 38,
    #     #             40, 41, 42, 43, 45, 46, 47, 48]
    # else:
    #     t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8,9]

    #loadpath = 'Data/T1wImgsMat' + SubjectName + '.mat'
    data = loadmat(params.TestingInvivoProspective)
    # data = loadmat(params.TestingPhantom)
    # data = loadmat(params.TestingSimPrecontrast)
    # data = loadmat(params.TestingSimPostcontrast)
    gBS=25 ##25 for prospective in-vivo, 45 for phantom_20210810; 116 for pre-contrast simulation and 108 for post-contrast simulation
    #load prospective pre-contrast testing dataset
    try:
        Pre5HBsT1wTIs_in = data['Pre5HBsT1wTIs']
        Pre5HBsT1wTIs_double = Pre5HBsT1wTIs_in.astype(np.double)
        Pre5HBs_tst_t1w_TI = np.abs(Pre5HBsT1wTIs_double[:,:,t1wtiIdx,:,:])

        tst_Len = Pre5HBs_tst_t1w_TI.shape[0]

        Pre5HBsMyoMask_in = data['Pre5HBsMyoMask']
        Pre5HBsMyoMask_double = Pre5HBsMyoMask_in.astype(np.double)
        Pre5HBsBPMask_in = data['Pre5HBsBPMask']
        Pre5HBsBPMask_double = Pre5HBsBPMask_in.astype(np.double)
        Pre5HBsSepMask_in = data['Pre5HBsSepMask']
        Pre5HBsSepMask_double = Pre5HBsSepMask_in.astype(np.double)


        Pre5HBsMyoMask = np.zeros((Pre5HBsMyoMask_double.shape[0], 1, Pre5HBsMyoMask_double.shape[1], Pre5HBsMyoMask_double.shape[2]))
        Pre5HBsBPMask = np.zeros((Pre5HBsBPMask_double.shape[0], 1, Pre5HBsBPMask_double.shape[1], Pre5HBsBPMask_double.shape[2]))
        Pre5HBsSepMask = np.zeros((Pre5HBsSepMask_double.shape[0], 1, Pre5HBsSepMask_double.shape[1], Pre5HBsSepMask_double.shape[2]))

        for ix in range(0, Pre5HBsMyoMask_double.shape[0]):
            Pre5HBsMyoMask[ix, 0, :, :] = Pre5HBsMyoMask_double[ix, :, :]
            Pre5HBsBPMask[ix, 0, :, :] = Pre5HBsBPMask_double[ix, :, :]
            Pre5HBsSepMask[ix, 0, :, :] = Pre5HBsSepMask_double[ix, :, :]

    except Exception as e:
        print(e)

    #load propective post-contrast testing dataset
    try:
        Post5HBsT1wTIs_in = data['Post5HBsT1wTIs']
        Post5HBsT1wTIs_double = Post5HBsT1wTIs_in.astype(np.double)
        Post5HBs_tst_t1w_TI = np.abs(Post5HBsT1wTIs_double[:,:,t1wtiIdx,:,:])

        if tst_Len <Post5HBs_tst_t1w_TI.shape[0]:
            tst_Len = Post5HBs_tst_t1w_TI.shape[0]

        Post5HBsMyoMask_in = data['Post5HBsMyoMask']
        Post5HBsMyoMask_double = Post5HBsMyoMask_in.astype(np.double)
        Post5HBsBPMask_in = data['Post5HBsBPMask']
        Post5HBsBPMask_double = Post5HBsBPMask_in.astype(np.double)
        Post5HBsSepMask_in = data['Post5HBsSepMask']
        Post5HBsSepMask_double = Post5HBsSepMask_in.astype(np.double)

        Post5HBsMyoMask = np.zeros((Post5HBsMyoMask_double.shape[0],1,Post5HBsMyoMask_double.shape[1], Post5HBsMyoMask_double.shape[2]))
        Post5HBsBPMask = np.zeros((Post5HBsBPMask_double.shape[0],1,Post5HBsBPMask_double.shape[1], Post5HBsBPMask_double.shape[2]))
        Post5HBsSepMask = np.zeros(
            (Post5HBsSepMask_double.shape[0], 1, Post5HBsSepMask_double.shape[1], Post5HBsSepMask_double.shape[2]))
        for ix in range(0, Post5HBsBPMask_double.shape[0]):
            Post5HBsMyoMask[ix,0,:,:] = Post5HBsMyoMask_double[ix,:,:]
            Post5HBsBPMask[ix,0,:,:] = Post5HBsBPMask_double[ix,:,:]
            Post5HBsSepMask[ix,0,:,:] = Post5HBsSepMask_double[ix,:,:]

    except Exception as e:
        print(e)

    try:
        PreMOLLIT1wTIs_in = data['PreMOLLIT1wTIs']
        PreMOLLIT1wTIs_double = PreMOLLIT1wTIs_in.astype(np.double)
        PreMOLLIT1wTIs = np.abs(PreMOLLIT1wTIs_double[:,:,t1wtiIdx,:,:])

        tst_Len = PreMOLLIT1wTIs.shape[0]

        PreMOLLIMyoMask_in = data['PreMOLLIMyoMask']
        PreMOLLIMyoMask_double = PreMOLLIMyoMask_in.astype(np.double)
        PreMOLLIBPMask_in = data['PreMOLLIBPMask']
        PreMOLLIBPMask_double = PreMOLLIBPMask_in.astype(np.double)
        PreMOLLISepMask_in = data['PreMOLLISepMask']
        PreMOLLISepMask_double = PreMOLLISepMask_in.astype(np.double)

        PreMOLLIMyoMask = np.zeros((PreMOLLIMyoMask_double.shape[0],1,PreMOLLIMyoMask_double.shape[1], PreMOLLIMyoMask_double.shape[2]))
        PreMOLLIBPMask = np.zeros((PreMOLLIBPMask_double.shape[0],1,PreMOLLIBPMask_double.shape[1], PreMOLLIBPMask_double.shape[2]))
        PreMOLLISepMask = np.zeros(
            (PreMOLLISepMask_double.shape[0], 1, PreMOLLISepMask_double.shape[1], PreMOLLISepMask_double.shape[2]))
        for ix in range(0, PreMOLLIMyoMask_double.shape[0]):
            PreMOLLIMyoMask[ix,0,:,:] = PreMOLLIMyoMask_double[ix,:,:]
            PreMOLLIBPMask[ix,0,:,:] = PreMOLLIBPMask_double[ix,:,:]
            PreMOLLISepMask[ix,0,:,:] = PreMOLLISepMask_double[ix,:,:]
    except Exception as e:
        print(e)

    try:
        PostMOLLIT1wTIs_in = data['PostMOLLIT1wTIs']
        PostMOLLIT1wTIs_double = np.abs(PostMOLLIT1wTIs_in.astype(np.double))
        PostMOLLIT1wTIs = PostMOLLIT1wTIs_double[:,:,t1wtiIdx,:,:]

        PostMOLLIMyoMask_in = data['PostMOLLIMyoMask']
        PostMOLLIMyoMask_double = PostMOLLIMyoMask_in.astype(np.double)
        PostMOLLIBPMask_in = data['PostMOLLIBPMask']
        PostMOLLIBPMask_double = PostMOLLIBPMask_in.astype(np.double)
        PostMOLLISepMask_in = data['PostMOLLISepMask']
        PostMOLLISepMask_double = PostMOLLISepMask_in.astype(np.double)

        PostMOLLIMyoMask = np.zeros((PostMOLLIMyoMask_double.shape[0],1,PostMOLLIMyoMask_double.shape[1], PostMOLLIMyoMask_double.shape[2]))
        PostMOLLIBPMask = np.zeros((PostMOLLIBPMask_double.shape[0],1,PostMOLLIBPMask_double.shape[1], PostMOLLIBPMask_double.shape[2]))
        PostMOLLISepMask = np.zeros(
            (PostMOLLISepMask_double.shape[0], 1, PostMOLLISepMask_double.shape[1], PostMOLLISepMask_double.shape[2]))
        for ix in range(0, PostMOLLIBPMask_double.shape[0]):
            PostMOLLIMyoMask[ix,0,:,:] = PostMOLLIMyoMask_double[ix,:,:]
            PostMOLLIBPMask[ix,0,:,:] = PostMOLLIBPMask_double[ix,:,:]
            PostMOLLISepMask[ix,0,:,:] = PostMOLLISepMask_double[ix,:,:]

    except Exception as e:
        print(e)

    # try:
    #     PreMOLLIT1MapOnLine_in = data['PreMOLLIT1MapOnLine']
    #     PreMOLLIT1MapOnLine_double = PreMOLLIT1MapOnLine_in.astype(np.double)
    #     PreMOLLIT1MapOnLineT1 = np.zeros((PreMOLLIT1MapOnLine_double.shape[0], 1, PreMOLLIT1MapOnLine_double.shape[1], PreMOLLIT1MapOnLine_double.shape[2]))
    #
    #     # tr_mask = np.ones(tr_T1.shape)
    #     for ix in range(0, PreMOLLIT1MapOnLine_double.shape[0]):
    #         PreMOLLIT1MapOnLineT1[ix, 0, :, :] =PreMOLLIT1MapOnLine_double[ix, :, :]
    # except Exception as e:
    #     print(e)
    #
    try:
        PreMOLLIT1MapOffLine_in = data['PreMOLLIT1MapOffLine']
        PreMOLLIT1MapOffLine_double = PreMOLLIT1MapOffLine_in.astype(np.double)
        PreMOLLIT1MapOffLineT1 = np.zeros((PreMOLLIT1MapOffLine_double.shape[0], 1, PreMOLLIT1MapOffLine_double.shape[1], PreMOLLIT1MapOffLine_double.shape[2]))
        # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PreMOLLIT1MapOffLine_double.shape[0]):
            PreMOLLIT1MapOffLineT1[ix, 0, :, :] = PreMOLLIT1MapOffLine_double = PreMOLLIT1MapOffLine_in.astype(np.double)[ix, :, :]
    except Exception as e:
        print(e)

    # try:
    #     PostMOLLIT1MapOnLine_in = data['PostMOLLIT1MapOnLine']
    #     PostMOLLIT1MapOnLine_double = PostMOLLIT1MapOnLine_in.astype(np.double)
    #     PostMOLLIT1MapOnLineT1 = np.zeros((PostMOLLIT1MapOnLine_double.shape[0], 1, PostMOLLIT1MapOnLine_double.shape[1], PostMOLLIT1MapOnLine_double.shape[2]))
    # # tr_mask = np.ones(tr_T1.shape)
    #     for ix in range(0, PostMOLLIT1MapOnLine_double.shape[0]):
    #         PostMOLLIT1MapOnLineT1[ix, 0, :, :] =PostMOLLIT1MapOnLine_double[ix, :, :]
    # except Exception as e:
    #     print(e)

    try:
        PostMOLLIT1MapOffLine_in = data['PostMOLLIT1MapOffLine']
        PostMOLLIT1MapOffLine_double = PostMOLLIT1MapOffLine_in.astype(np.double)
        PostMOLLIT1MapOffLineT1 = np.zeros((PostMOLLIT1MapOffLine_double.shape[0], 1, PostMOLLIT1MapOffLine_double.shape[1], PostMOLLIT1MapOffLine_double.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PostMOLLIT1MapOffLine_double.shape[0]):
            PostMOLLIT1MapOffLineT1[ix, 0, :, :] =PostMOLLIT1MapOffLine_double[ix, :, :]
    except Exception as e:
        print(e)



    try:
        PreSASHA2PT1MapOffLine_in = data['PreSASHA2PT1MapOffLine']
        PreSASHA2PT1MapOffLine_double = PreSASHA2PT1MapOffLine_in.astype(np.double)
        PreSASHA2PT1MapOffLineT1 = np.zeros((PreSASHA2PT1MapOffLine_double.shape[0], 1, PreSASHA2PT1MapOffLine_double.shape[1], PreSASHA2PT1MapOffLine_double.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PreSASHA2PT1MapOffLine_double.shape[0]):
            PreSASHA2PT1MapOffLineT1[ix, 0, :, :] =PreSASHA2PT1MapOffLine_double[ix, :, :]
    except Exception as e:
        print(e)

    try:
        PreSASHA3PT1MapOffLine_in = data['PreSASHA3PT1MapOffLine']
        PreSASHA3PT1MapOffLine_double = PreSASHA3PT1MapOffLine_in.astype(np.double)
        PreSASHA3PT1MapOffLineT1 = np.zeros((PreSASHA3PT1MapOffLine_double.shape[0], 1, PreSASHA3PT1MapOffLine_double.shape[1], PreSASHA3PT1MapOffLine_double.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PreSASHA3PT1MapOffLine_double.shape[0]):
            PreSASHA3PT1MapOffLineT1[ix, 0, :, :] =PreSASHA3PT1MapOffLine_double[ix, :, :]
    except Exception as e:
        print(e)

    try:
        PreSASHAMyoMask_in = data['PreSASHAMyoMask']
        PreSASHAMyoMask_double = PreSASHAMyoMask_in.astype(np.double)
        PreSASHABPMask_in = data['PreSASHABPMask']
        PreSASHABPMask_double = PreSASHABPMask_in.astype(np.double)
        PreSASHASepMask_in = data['PreSASHASepMask']
        PreSASHASepMask_double = PreSASHASepMask_in.astype(np.double)

        PreSASHAMyoMask = np.zeros(
            (PreSASHAMyoMask_double.shape[0], 1, PreSASHAMyoMask_double.shape[1], PreSASHAMyoMask_double.shape[2]))
        PreSASHABPMask = np.zeros(
            (PreSASHABPMask_double.shape[0], 1, PreSASHABPMask_double.shape[1], PreSASHABPMask_double.shape[2]))
        PreSASHASepMask = np.zeros(
            (PreSASHASepMask_double.shape[0], 1, PreSASHASepMask_double.shape[1], PreSASHASepMask_double.shape[2]))

        for ix in range(0, PreSASHABPMask_double.shape[0]):
            PreSASHAMyoMask[ix, 0, :, :] = PreSASHAMyoMask_double[ix, :, :]
            PreSASHABPMask[ix, 0, :, :] = PreSASHABPMask_double[ix, :, :]
            PreSASHASepMask[ix, 0, :, :] = PreSASHASepMask_double[ix, :, :]
    except Exception as e:
        print(e)

    try:
        PostSASHA2PT1MapOffLine_in = data['PostSASHA2PT1MapOffLine']
        PostSASHA2PT1MapOffLine_double = PostSASHA2PT1MapOffLine_in.astype(np.double)
        PostSASHA2PT1MapOffLineT1 = np.zeros((PostSASHA2PT1MapOffLine_double.shape[0], 1,
                                             PostSASHA2PT1MapOffLine_double.shape[1],
                                             PostSASHA2PT1MapOffLine_double.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PostSASHA2PT1MapOffLine_double.shape[0]):
            PostSASHA2PT1MapOffLineT1[ix, 0, :, :] = PostSASHA2PT1MapOffLine_double[ix, :, :]
    except Exception as e:
        print(e)

    try:
        PostSASHA3PT1MapOffLine_in = data['PostSASHA3PT1MapOffLine']
        PostSASHA3PT1MapOffLine_double = PostSASHA3PT1MapOffLine_in.astype(np.double)
        PostSASHA3PT1MapOffLineT1 = np.zeros((PostSASHA3PT1MapOffLine_double.shape[0], 1,
                                             PostSASHA3PT1MapOffLine_double.shape[1],
                                             PostSASHA3PT1MapOffLine_double.shape[2]))
        # # tr_mask = np.ones(tr_T1.shape)
        for ix in range(0, PostSASHA3PT1MapOffLine_double.shape[0]):
            PostSASHA3PT1MapOffLineT1[ix, 0, :, :] = PostSASHA3PT1MapOffLine_double[ix, :, :]
    except Exception as e:
        print(e)

    try:
        PostSASHAMyoMask_in = data['PostSASHAMyoMask']
        PostSASHAMyoMask_double = PostSASHAMyoMask_in.astype(np.double)
        PostSASHABPMask_in = data['PostSASHABPMask']
        PostSASHABPMask_double = PostSASHABPMask_in.astype(np.double)
        PostSASHASepMask_in = data['PostSASHASepMask']
        PostSASHASepMask_double = PostSASHASepMask_in.astype(np.double)

        PostSASHAMyoMask = np.zeros(
            (PostSASHAMyoMask_double.shape[0], 1, PostSASHAMyoMask_double.shape[1], PostSASHAMyoMask_double.shape[2]))
        PostSASHABPMask = np.zeros(
            (PostSASHABPMask_double.shape[0], 1, PostSASHABPMask_double.shape[1], PostSASHABPMask_double.shape[2]))
        PostSASHASepMask = np.zeros(
            (PostSASHASepMask_double.shape[0], 1, PostSASHASepMask_double.shape[1], PostSASHASepMask_double.shape[2]))

        for ix in range(0, PostSASHABPMask_double.shape[0]):
            PostSASHAMyoMask[ix, 0, :, :] = PostSASHAMyoMask_double[ix, :, :]
            PostSASHABPMask[ix, 0, :, :] = PostSASHABPMask_double[ix, :, :]
            PostSASHASepMask[ix, 0, :, :] = PostSASHASepMask_double[ix, :, :]
    except Exception as e:
        print(e)


    try:
        subjectsLists = data['subjectsLists']
        subjectsLists = subjectsLists.tolist()

    except Exception as e:
        print(e)

    # plt.imshow(PreSASHA3PT1MapOnLine[1,0,:,:], cmap='jet', vmin=0, vmax=1.2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 map")
    # plt.show()

#-------------------------------------------------------------#
#-----------------------Training------------------------------#
#-------------------------------------------------------------#

def train(net):

    ##
    i = itNum  #
    # fig, axs = plt.subplots(5, 6)
    params.batch_size = 20  # Batch size is 40 for simulation and in-vivo
    tr_N = tr_T1.shape[0]
    tr_lst = list(range(0, tr_N))

    initialLoss = 1e10
    bestModelEpochIx = 0
    tmpMyoBloodLoss = 0
    trainLossLst = list()

    DebugSave = False
    #epoch loop
    for epoch in range(s_epoch, params.epochs+1):
        print('epoch {}/{}...'.format(epoch, params.epochs))

        random.shuffle(tr_lst)  # This is only used for in vivo data

        try:
            ## start Training
            l = 0
            itt = 0
            TAG = 'Training'
            MAX = list()

            #batch loop
            for idx in range(0, tr_N, params.batch_size):
                try:
                    lst = tr_lst[idx:idx+params.batch_size]
                    X = Variable(torch.FloatTensor(tr_t1w_TI[lst,:,:,:,:])).to('cuda:0')
                    y = Variable(torch.FloatTensor(tr_T1[lst,:,:,:])).to('cuda:0')
                    # T1_5 = Variable(torch.FloatTensor(tr_T1_5[lst,:,:,:])).to('cuda:0')
                    w_mask = Variable(torch.FloatTensor(tr_mask[lst,:,:,:])).to('cuda:0')
                    #sliceID = tr_sliceID[lst].tolist()


                    xs = X.shape
                    X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0]*xs[3]*xs[4],xs[1],xs[2]))
                    y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                    w_mask = w_mask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                    #predicated by net
                    net.train()
                    y_pred = net(X.to('cuda:0')).to('cuda:0')

                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    continue

                # if i > 0 and i % 1000 == 0 and idx ==0:
                #     ifnames = ['Image_epoch_{0}_iter_{1}_sl_{2}'.format(epoch, i, s) for s in range(0, params.batch_size)]
                #     # T1_orig = Variable(torch.FloatTensor(T1.float()))
                #     #fignames = ['ptn{0}_{1}'.format(s.split('/')[-2][:], s.split('/')[-1][:-5]) for s in sliceID]
                #     # nuft = torch.ifft(X[:,:,params.moving_window_size//2,:,:,:].squeeze(2), 2, normalized=True)   #fftshift2d(torch.ifft(X, 2), [2, 3])
                #     diffPredRef = torch.abs(y-y_pred)
                #     ntensorshow((T1_5, y_pred.reshape((xs[0],1,xs[3],xs[4])), y.reshape((xs[0],1,xs[3],xs[4])),
                #                  diffPredRef.reshape((xs[0],1,xs[3],xs[4]))), (0, 0), (0, 4), ('Fitting 5T1w',  'Net', 'MOLLI (Ref)','Net-MOLLI'),
                #                 saveFigs = True, figname=ifnames)

                # if epoch < 500:
                #     loss = w_mse(y_pred, y, w_mask)
                # else:
                #     loss = w_mae(y_pred, y, w_mask)
                # loss = l1Criterion(magnitude(y_pred), y)
                # loss = l1Criterion(y_pred, y)
                # loss = mseCriterion(torch.log(magnitude(y_pred) + 1), torch.log(magnitude(y) + 1))

                #for 4HBsPreLowLR and 4HBsPreandPost
                # if epoch > 1500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.05] = 0
                # if epoch > 2000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.03] = 0

                # #for 4HBsPre33
                # if epoch > 500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.2] = 0
                # if epoch > 1000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.1] = 0

                #for all four models
                # if epoch > 2000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.2] = 0
                # if epoch > 2500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.1] = 0

                # if epoch > 5000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.05] = 0
                # if epoch > 5500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.03] = 0
                loss = w_mae(y_pred, y, w_mask)

                if DebugSave:
                    predT1 = y_pred.reshape((xs[0], 1, xs[3], xs[4]))
                    refT1 = y.reshape((xs[0], 1, xs[3], xs[4]))
                    maskSave = w_mask.reshape((xs[0], 1, xs[3], xs[4]))
                    saveArrayToMat(predT1.cpu().data.numpy(), 'predT1',
                                   'predT1forDebug',
                                   params.tensorboard_dir)
                    saveArrayToMat(refT1.cpu().data.numpy(), 'refT1',
                                   'refT1forDebug',
                                   params.tensorboard_dir)
                    saveArrayToMat(maskSave.cpu().data.numpy(), 'mask',
                                   'maskforDebug',
                                   params.tensorboard_dir)

                LOSS.append(loss.cpu().data.numpy())

                l += loss.data

                optimizer.zero_grad()
                loss.backward()
                i += 1
                optimizer.step()
                # torch.nn.utils.clip_grad_norm_(net.parameters(),0.25)

                print('Epoch: {0} - {1:.3f}%'.format(epoch, 100 * (itt * params.batch_size) /tr_N)
                      + ' \tIter: ' + str(i)
                      + '\tLoss: {0:.6f}'.format(loss.data)
                      # + '\tInputLoss: {0:.6f}'.format(inloss.data[0])
                      )
                itt += 1
                is_best = 0

                #trained model is backed up every 50 iteration
                if i % 50 == 0:
                    save_checkpoint({'epoch': epoch, 'loss': LOSS, 'arch': 'recoNet_Model1', 'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'iteration': i,
                                    }, is_best, filename=params.model_save_dir + 'MODEL_EPOCH{}.pth'.format(epoch))

                # if True or params.tbVisualize:
                #     writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
                #     saveArrayToMat(LOSS, 'mse',
                #                    'mse_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)

            avg_loss = params.batch_size * l / tr_N
            print('Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(l, avg_loss))

            save_checkpoint({'epoch': epoch, 'loss': LOSS, 'arch': 'recoNet_Model1', 'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict(), 'iteration': i,
                             }, is_best, filename=params.model_save_dir + 'MODEL_EPOCH{}.pth'.format(epoch))

            trainLossLst.append(avg_loss.cpu().data.numpy())
            # writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
            # saveArrayToMat(LOSS, 'mse',
            #                'mse_R{0}_Trial{1}_Train'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)

            ##validation Code
            T1_5_avg = list()
            ref_T1_avg = list()
            pred_T1_5_avg = list()

            allMyoBPPixelsPredict = list()
            allMyoBPPixelsRef = list()

            meanT1Ref = list()
            meanT1Pre = list()

            meanMyoT1Ref = list()       #Reference
            meanMyoT1Pre = list()       #Predicted by Net
            meanBloodT1Ref = list()     #Reference
            meanBloodT1Pre = list()     #Predicted by Net
            val_N = val_t1w_TI.shape[0]
            val_lst = list(range(0, val_N))
            sl_id = list()
            bs = val_N ##
            save_PNG = False

            myoLossLst = list()
            bpLossLst = list()
            myLossTotal = 0
            bpLossTotal = 0
            TAG = 'Validation'

            with torch.no_grad():
                for idx in range(0, val_N, bs):
                    # for X, y, T1, TI, T1_5, mask, LVmask, sliceID in training_DG:
                    try:

                        X = Variable(torch.FloatTensor(val_t1w_TI[val_lst[idx:idx + bs]])).to('cuda:0')
                        y = Variable(torch.FloatTensor(val_T1[val_lst[idx:idx + bs]])).to('cuda:0')
                        T1_5 = Variable(torch.FloatTensor(val_T1[val_lst[idx:idx + bs]])).to('cuda:0')
                        # LVmask = Variable(torch.FloatTensor(tst_LVmask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # ROImask = Variable(torch.FloatTensor(tst_ROImask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # bloodmask = Variable(torch.FloatTensor(tst_mask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # sliceID = tst_sliceID[np.array(val_lst[idx:idx + bs])].tolist()
                        # LVBpmask = Variable(torch.FloatTensor(val_LVBpmask[val_lst[idx:idx + bs]])).to('cuda:0')

                        myomask = Variable(torch.FloatTensor(val_Myomask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # bpmask = Variable(torch.FloatTensor(val_BPmask[val_lst[idx:idx + bs]])).to('cuda:0')

                        # LVBpmask = tst_LVBpmask[val_lst[idx:idx + bs]]
                        # t_const = 1e6
                        # X = torch.cat(
                        #     (X[:, :, 0:5, :, :, 0] * t_const, X[:, :, 0:5, :, :, 1] * t_const, X[:, :, 5:, :, :, 0]), 2)
                        # X = torch.cat((magnitude(X[:, :, 0:5, :, :, :]) * t_const, X[:, :, 5:, :, :, 0]), 2)
                        xs = X.shape
                        X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # TI = TI.unsqueeze(1).permute((0,3,4,1,2)).reshape((xs[0]*xs[3]*xs[4],xs[1],xs[2]))
                        # X = torch.cat((X*t_const, TI), 2)
                        # w_mask = w_mask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # LVBpmask = LVBpmask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        myoMaskLoss = myomask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # bpMaskLoss = bpmask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                        net.eval()
                        y_pred = net(X.to('cuda:0')).to('cuda:0')

                        pred_T1_5 = y_pred.reshape((xs[0], 1, xs[3], xs[4]))
                        ref_T1 = y.reshape((xs[0], 1, xs[3], xs[4]))
                        # LVBpmask = LVBpmask.reshape((xs[0], 1, xs[3], xs[4]))
                        # allMyoBPPixelsPredict.append(y_pred[np.nonzero(LVBpmask.cpu().data.numpy())].cpu().data.numpy())
                        # allMyoBPPixelsRef.append(y[np.nonzero(LVBpmask.cpu().data.numpy())].cpu().data.numpy())




                        #loss regarding myocrdium and blood pool

                        # if epoch > 1500:
                        #     losstmp = myoMaskLoss * torch.abs(y_pred - y)
                        #     myoMaskLoss[losstmp > 0.2] = 0
                        #     # losstmp = bpMaskLoss * torch.abs(y_pred - y)
                        #     # bpMaskLoss[losstmp > 0.2] = 0
                        #
                        # if epoch > 2000:
                        #     losstmp = myoMaskLoss * torch.abs(y_pred - y)
                        #     myoMaskLoss[losstmp > 0.1] = 0
                            # losstmp = bpMaskLoss * torch.abs(y_pred - y)
                            # bpMaskLoss[losstmp > 0.1] = 0
                        # if epoch > 3000:
                        #     losstmp = myoMaskLoss * torch.abs(y_pred - y)
                        #     myoMaskLoss[losstmp > 0.2] = 0
                        #     losstmp = bpMaskLoss * torch.abs(y_pred - y)
                        #     bpMaskLoss[losstmp > 0.2] = 0
                        #here, myoLoss is for simulation data
                        myoLossTmp = w_mae(y_pred[0:sim_Val_N*xs[3]*xs[4]-1,0], y[0:sim_Val_N*xs[3]*xs[4]-1,0], myoMaskLoss[0:sim_Val_N*xs[3]*xs[4]-1,0])
                        myoLossLst.append(myoLossTmp.cpu().data.numpy())
                        myLossTotal+=myoLossTmp.data*bs
                        #here, bpLoss is use for calculating loss of phantom data
                        bpLossTmp = w_mae(y_pred[sim_Val_N*xs[3]*xs[4]:,0], y[sim_Val_N*xs[3]*xs[4]:,0], myoMaskLoss[sim_Val_N*xs[3]*xs[4]:,0])
                        bpLossLst.append(bpLossTmp.cpu().data.numpy())
                        bpLossTotal+=bpLossTmp.data*bs

                        if False:
                            predT1 = y_pred.reshape((xs[0], 1, xs[3], xs[4]))
                            refT1 = y.reshape((xs[0], 1, xs[3], xs[4]))
                            myomaskSave = myoMaskLoss.reshape((xs[0], 1, xs[3], xs[4]))
                            bpMaskSave = bpMaskLoss.reshape((xs[0], 1, xs[3], xs[4]))
                            saveArrayToMat(predT1.cpu().data.numpy(), 'predT1',
                                           'predT1forDebug',
                                           params.validation_dir)
                            saveArrayToMat(refT1.cpu().data.numpy(), 'refT1',
                                           'refT1forDebug',
                                           params.validation_dir)
                            saveArrayToMat(myomaskSave.cpu().data.numpy(), 'myomaskSave',
                                           'myomaskSave',
                                           params.validation_dir)
                            saveArrayToMat(bpMaskSave.cpu().data.numpy(), 'bpMaskSave',
                                           'bpMaskSave',
                                           params.validation_dir)
                        if True:
                            ##save image
                            try:
                                n_slices = pred_T1_5.shape[0]

                                for sl in range(0, n_slices):
                                    T1tmp = ref_T1[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # #T1tmp = T1tmp[~np.isnan(T1tmp)]
                                    # meanT1Ref.append(T1tmp.mean())
                                    #
                                    # T1tmp = pred_T1_5[sl, 0, :, :].cpu().data.numpy() * LVBpmask[sl, 0, :,
                                    #                                                     :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # meanT1Pre.append(T1tmp.mean())
                                    #
                                    # ##calculate Myo T1
                                    # T1tmp = ref_T1[sl, 0, :, :].cpu().data.numpy() * myomask[sl, 0, :,
                                    #                                                  :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # meanMyoT1Ref.append(T1tmp.mean())
                                    # T1tmp = pred_T1_5[sl, 0, :, :].cpu().data.numpy() * myomask[sl, 0, :,
                                    #                                                     :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # meanMyoT1Pre.append(T1tmp.mean())
                                    #
                                    #
                                    # ##calcualte blood pool T1
                                    # T1tmp = ref_T1[sl, 0, :, :].cpu().data.numpy() * bpmask[sl, 0, :,
                                    #                                                  :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # meanBloodT1Ref.append(T1tmp.mean())
                                    # T1tmp = pred_T1_5[sl, 0, :, :].cpu().data.numpy() * bpmask[sl, 0, :,
                                    #                                                     :].cpu().data.numpy() * TimeScaling
                                    # T1tmp = T1tmp[T1tmp > 0]
                                    # meanBloodT1Pre.append(T1tmp.mean())

                                    #for checking the code
                                    #         # plt.imshow(tst_t1w_TI[ix,0,5,:,:], cmap='plasma', vmin=0, vmax=1)
                                    #         # plt.colorbar()
                                    #         # plt.xticks(())
                                    #         # plt.yticks(())
                                    #         # plt.title("T1 maps")
                                    #         # plt.show()

                                    if save_PNG or epoch== params.epochs or epoch==100 or epoch ==500 or epoch == 1000:
                                        fig, axs = plt.subplots(1, 3)
                                        axs[0].set_title('Net')
                                        axs[0].imshow(pred_T1_5[sl, 0, :, :].cpu().data.numpy() * TimeScaling, cmap='jet',
                                                      vmin=0, vmax=2500)
                                        # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                                        axs[0].axis('off')

                                        axs[1].set_title('MOLLI53')
                                        axs[1].imshow(ref_T1[sl, 0, :, :].cpu().data.numpy() * TimeScaling, cmap='jet', vmin=0,
                                                      vmax=2500)
                                        # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                                        axs[1].axis('off')

                                        axs[2].set_title('Net-MOLLI53')
                                        axs[2].imshow(np.abs(pred_T1_5[sl, 0, :, :].cpu().data.numpy() - ref_T1[sl, 0, :,
                                                                                                         :].cpu().data.numpy()) * TimeScaling,
                                                      cmap='jet', vmin=0, vmax=2500)
                                        # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                                        axs[2].axis('off')

                                        # fig.show()

                                        # fig.savefig(params.validation_dir + tst_sliceID[sl] + '.png')
                                        fig.savefig(params.validation_dir + str(sl) + '.png')
                                        plt.close(fig)
                            except Exception as e:
                                traceback.print_exc()
                                continue

                    except Exception as e:
                        traceback.print_exc()
                        continue
                avg_myo_loss = myLossTotal/val_N
                myoAvgLossAllEpochs.append(avg_myo_loss.cpu().data.numpy())
                print('Sim: Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(myLossTotal, avg_myo_loss))
                avg_bp_loss = bpLossTotal / val_N
                bpAvgLossAllEpochs.append(avg_bp_loss.cpu().data.numpy())
                print('Phantom: Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(bpLossTotal, avg_bp_loss))

                tmpMyoBloodLoss =avg_myo_loss+avg_bp_loss
                if initialLoss > tmpMyoBloodLoss:
                    initialLoss = tmpMyoBloodLoss
                    bestModelEpochIx = epoch

                print('The best model is @ epoch: {0:.0f} \t with  Total Sim+phantom Loss: {1:.6f}'.format(bestModelEpochIx, initialLoss))

                # if epoch == params.epochs:
                #     #saveArrayToMat(sl_id, 'sl_id')
                #     ##save myo T1
                #     saveArrayToMat(np.array(meanMyoT1Ref), 'meanMyoT1Ref', 'MeanRefMyoT1_R{0}_Trial{1}_Epoch_{2}'.format(str(params.Rate), params.trialNum,epoch), params.validation_dir)
                #     saveArrayToMat(np.array(meanMyoT1Pre), 'meanMyoT1Pre','MeanPreMyoT1_R{0}_Trial{1}_Epoch_{2}'.format(str(params.Rate), params.trialNum,epoch),params.validation_dir)
                #     ##save blood T1
                #     saveArrayToMat(np.array(meanBloodT1Ref), 'meanBloodT1Ref', 'MeanRefBPT1_R{0}_Trial{1}_Epoch_{2}'.format(str(params.Rate), params.trialNum,epoch),params.validation_dir)
                #     saveArrayToMat(np.array(meanMyoT1Pre), 'meanMyoT1Pre','MeanPreBPT1_R{0}_Trial{1}_Epoch_{2}'.format(str(params.Rate), params.trialNum,epoch) ,params.validation_dir)

        except Exception as e:
            traceback.print_exc()
            # print(e)
            continue

        except KeyboardInterrupt:
            print('Interrupted')
            torch.save(MyoMapNet.state_dict(), 'MODEL_INTERRUPTED.pth')
            saveArrayToMat(np.array(myoAvgLossAllEpochs), 'SimAvgLossAllEpochs',
                           'avgLoss_Sim_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),
                           params.validation_dir)
            saveArrayToMat(np.array(bpAvgLossAllEpochs), 'PhantomAvgLossAllEpochs',
                           'avgLoss_Phantom_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),
                           params.validation_dir)
            # writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
            saveArrayToMat(np.array(trainLossLst), 'mse',
                           'mse_R{0}_Trial{1}_train'.format(str(params.Rate), params.trialNum), params.validation_dir)

            saveArrayToMat(bestModelEpochIx, 'BestEpochIx',
                           'BestEpochIx_R{0}_Trial{1}_BestModelAtEpoch_{2}'.format(str(params.Rate), params.trialNum,
                                                                                   bestModelEpochIx),
                           params.validation_dir)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    saveArrayToMat(np.array(myoAvgLossAllEpochs), 'SimAvgLossAllEpochs','avgLoss_Sim_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),params.validation_dir)
    saveArrayToMat(np.array(bpAvgLossAllEpochs), 'PhantomAvgLossAllEpochs','avgLoss_Phantom_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),params.validation_dir)
    saveArrayToMat(np.array(trainLossLst), 'mse',
                   'mse_R{0}_Trial{1}_train'.format(str(params.Rate), params.trialNum), params.validation_dir)
    saveArrayToMat(bestModelEpochIx, 'BestEpochIx',
                   'BestEpochIx_R{0}_Trial{1}_BestModelAtEpoch_{2}'.format(str(params.Rate), params.trialNum,bestModelEpochIx), params.validation_dir)

    # writer.close()

#-----------------------------------------------------------#
#---------------------Testing----------------------------#
#-----------------------------------------------------------#

def Testing(net):

    Pre5HBsPredMyoMeanandStdLst = [[],[]]
    Pre5HBsPredBPMeanandStdLst = [[],[]]
    Pre5HBsPredSepMeanandStdLst = [[],[]]

    Post5HBsPredMyoMeanandStdLst = [[],[]]
    Post5HBsPredBPMeanandStdLst = [[],[]]
    Post5HBsPredSepMeanandStdLst = [[], []]

    PreMOLLIPredMyoMeanandStdLst = [[],[]]
    PreMOLLIPredBPMeanandStdLst = [[],[]]
    PreMOLLIPredSepMeanandStdLst = [[], []]

    PostMOLLIPredMyoMeanandStdLst = [[],[]]
    PostMOLLIPredBPMeanandStdLst = [[],[]]
    PostMOLLIPredSepMeanandStdLst = [[], []]

    PreMOLLIOfflineMyoMeanandStdLst = [[],[]]
    PreMOLLIOfflineBPMeanandStdLst = [[],[]]
    PreMOLLIOfflineSepMeanandStdLst = [[], []]

    PostMOLLIOfflineMyoMeanandStdLst = [[],[]]
    PostMOLLIOfflineBPMeanandStdLst = [[],[]]
    PostMOLLIOfflineSepMeanandStdLst = [[], []]

    # PreMOLLIOnlineMyoMeanandStdLst = [[],[]]
    # PreMOLLIOnlineBPMeanandStdLst = [[],[]]
    #
    # PostMOLLIOnlineMyoMeanandStdLst = [[],[]]
    # PostMOLLIOnlineBPMeanandStdLst = [[],[]]

    PreSASHA2POfflineMyoMeanandStdLst = [[],[]]
    PreSASHA2POfflineBPMeanandStdLst = [[], []]
    PreSASHA2POfflineSepMeanandStdLst = [[], []]

    PreSASHA3POfflineMyoMeanandStdLst = [[],[]]
    PreSASHA3POfflineBPMeanandStdLst = [[], []]
    PreSASHA3POfflineSepMeanandStdLst = [[], []]

    PostSASHA2POfflineMyoMeanandStdLst = [[], []]
    PostSASHA2POfflineBPMeanandStdLst = [[], []]
    PostSASHA2POfflineSepMeanandStdLst = [[], []]

    PostSASHA3POfflineMyoMeanandStdLst = [[], []]
    PostSASHA3POfflineBPMeanandStdLst = [[], []]
    PostSASHA3POfflineSepMeanandStdLst = [[], []]

    PreSubjectsLists = list()
    PostSubjectsLists = list()

    allResults = {}

    tst_N = tst_Len
    tst_lst = list(range(0,tst_N))
    sl_id = list()
    bs =gBS

    TAG = 'Testing'
    net.eval()
    try:

        with torch.no_grad():
            for idx in range(0, tst_N, bs):
                # for X, y, T1, TI, T1_5, mask, LVmask, sliceID in training_DG:
                try:
                    preSubjectsLst = []
                    postSubjectsLst = []
                    # Prospective pre-contrast
                    try:
                        Pre5HBsDat = Variable(torch.FloatTensor(Pre5HBs_tst_t1w_TI[tst_lst[idx:idx + bs]])).to('cuda:0')
                        xs = Pre5HBsDat.shape
                        Pre5HBsDat = Pre5HBsDat.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        Pre5HBsPred = net(Pre5HBsDat.to('cuda:0')).to('cuda:0')
                        Pre5HBsPredT1 = Pre5HBsPred.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPre5HBs = Pre5HBsMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPre5HBs = Pre5HBsBPMask[tst_lst[idx:idx + bs]]
                        SepMaskPre5HBs = Pre5HBsSepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI( Pre5HBsPredT1*TimeScaling, MyoMaskPre5HBs)
                        Pre5HBsPredMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        Pre5HBsPredMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( Pre5HBsPredT1*TimeScaling, BPMaskPre5HBs)
                        Pre5HBsPredBPMeanandStdLst[0].extend(meanandstdT1[0])
                        Pre5HBsPredBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( Pre5HBsPredT1*TimeScaling, SepMaskPre5HBs)
                        Pre5HBsPredSepMeanandStdLst[0].extend(meanandstdT1[0])
                        Pre5HBsPredSepMeanandStdLst[1].extend(meanandstdT1[1])

                        for ix in range(0,bs):
                            ij = tst_lst[idx+ix]
                            preSubjectsLst.append(subjectsLists[ij])

                        PreSubjectsLists.extend(preSubjectsLst)

                    except Exception as e:
                        print(e)
                        # continue

                    # Prospective post-contrast
                    try:
                        Post5HBsDat = Variable(torch.FloatTensor(Post5HBs_tst_t1w_TI[tst_lst[idx:idx + bs]])).to('cuda:0')
                        xs = Post5HBsDat.shape
                        Post5HBsDat = Post5HBsDat.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        Post5HBsPred = net(Post5HBsDat.to('cuda:0')).to('cuda:0')
                        Post5HBsPredT1 = Post5HBsPred.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPost5HBs = Post5HBsMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPost5HBs = Post5HBsBPMask[tst_lst[idx:idx + bs]]
                        SepMaskPost5HBs = Post5HBsSepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI( Post5HBsPredT1*TimeScaling, MyoMaskPost5HBs)
                        Post5HBsPredMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        Post5HBsPredMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( Post5HBsPredT1*TimeScaling, BPMaskPost5HBs)
                        Post5HBsPredBPMeanandStdLst[0].extend(meanandstdT1[0])
                        Post5HBsPredBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( Post5HBsPredT1*TimeScaling, SepMaskPost5HBs)
                        Post5HBsPredSepMeanandStdLst[0].extend(meanandstdT1[0])
                        Post5HBsPredSepMeanandStdLst[1].extend(meanandstdT1[1])

                        for ix in range(0,bs):
                            ij = tst_lst[idx+ix]
                            postSubjectsLst.append(subjectsLists[ij])

                        PostSubjectsLists.extend(postSubjectsLst)

                    except Exception as e:
                        print(e)
                        # continue

                    # MOLLI pre-contrast
                    try:
                        PreMOLLI5HBsDat = Variable(torch.FloatTensor(PreMOLLIT1wTIs[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        xs = PreMOLLI5HBsDat.shape
                        PreMOLLI5HBsDat = PreMOLLI5HBsDat.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        PreMOLLI5HBsPred = net(PreMOLLI5HBsDat.to('cuda:0')).to('cuda:0')
                        PreMOLLI5HBsPredT1 = PreMOLLI5HBsPred.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPreMOLLI5HBs = PreMOLLIMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPreMOLLI5HBs = PreMOLLIBPMask[tst_lst[idx:idx + bs]]
                        SepMaskPreMOLLI5HBs = PreMOLLISepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI( PreMOLLI5HBsPredT1*TimeScaling, MyoMaskPreMOLLI5HBs)
                        PreMOLLIPredMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIPredMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( PreMOLLI5HBsPredT1*TimeScaling, BPMaskPreMOLLI5HBs)
                        PreMOLLIPredBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIPredBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreMOLLI5HBsPredT1 * TimeScaling, SepMaskPreMOLLI5HBs)
                        PreMOLLIPredSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIPredSepMeanandStdLst[1].extend(meanandstdT1[1])


                        PreMOLLIT1MapOFFLine = Variable(torch.FloatTensor(PreMOLLIT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PreMOLLIT1MapOFFLine = PreMOLLIT1MapOFFLine.reshape((xs[0], 1, xs[3], xs[4]))

                        meanandstdT1 = mean_std_ROI(PreMOLLIT1MapOFFLine*TimeScaling, MyoMaskPreMOLLI5HBs)
                        PreMOLLIOfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIOfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreMOLLIT1MapOFFLine*TimeScaling, BPMaskPreMOLLI5HBs)
                        PreMOLLIOfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIOfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreMOLLIT1MapOFFLine*TimeScaling, SepMaskPreMOLLI5HBs)
                        PreMOLLIOfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PreMOLLIOfflineSepMeanandStdLst[1].extend(meanandstdT1[1])

                    except Exception as e:
                        print(e)
                        # continue

                    try:
                        PostMOLLI5HBsDat = Variable(torch.FloatTensor(PostMOLLIT1wTIs[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        xs = PostMOLLI5HBsDat.shape
                        PostMOLLI5HBsDat = PostMOLLI5HBsDat.permute((0, 3, 4, 1, 2)).reshape(
                            (xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        PostMOLLI5HBsPred = net(PostMOLLI5HBsDat.to('cuda:0')).to('cuda:0')
                        PostMOLLI5HBsPredT1 = PostMOLLI5HBsPred.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPostMOLLI5HBs = PostMOLLIMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPostMOLLI5HBs = PostMOLLIBPMask[tst_lst[idx:idx + bs]]
                        SepMaskPostMOLLI5HBs = PostMOLLISepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI( PostMOLLI5HBsPredT1*TimeScaling, MyoMaskPostMOLLI5HBs)
                        PostMOLLIPredMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIPredMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( PostMOLLI5HBsPredT1*TimeScaling, BPMaskPostMOLLI5HBs)
                        PostMOLLIPredBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIPredBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( PostMOLLI5HBsPredT1*TimeScaling, SepMaskPostMOLLI5HBs)
                        PostMOLLIPredSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIPredSepMeanandStdLst[1].extend(meanandstdT1[1])

                        PostMOLLIT1MapOFFLine = Variable(torch.FloatTensor(PostMOLLIT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PostMOLLIT1MapOFFLine = PostMOLLIT1MapOFFLine.reshape((xs[0], 1, xs[3], xs[4]))

                        meanandstdT1 = mean_std_ROI( PostMOLLIT1MapOFFLine*TimeScaling, MyoMaskPostMOLLI5HBs)
                        PostMOLLIOfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIOfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( PostMOLLIT1MapOFFLine*TimeScaling, BPMaskPostMOLLI5HBs)
                        PostMOLLIOfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIOfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI( PostMOLLIT1MapOFFLine*TimeScaling, SepMaskPostMOLLI5HBs)
                        PostMOLLIOfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PostMOLLIOfflineSepMeanandStdLst[1].extend(meanandstdT1[1])

                    except Exception as e:
                        print(e)
                        # continue

                    try:
                        PreSASHA2PT1MapOnLine = Variable(
                            torch.FloatTensor(PreSASHA2PT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PreSASHA2PT1MapOnLine = PreSASHA2PT1MapOnLine.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPreSASHA = PreSASHAMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPreSASHA = PreSASHABPMask[tst_lst[idx:idx + bs]]
                        SepMaskPreSASHA = PreSASHASepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI(PreSASHA2PT1MapOnLine*TimeScaling, MyoMaskPreSASHA)
                        PreSASHA2POfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA2POfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreSASHA2PT1MapOnLine*TimeScaling, BPMaskPreSASHA)
                        PreSASHA2POfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA2POfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreSASHA2PT1MapOnLine*TimeScaling, SepMaskPreSASHA)
                        PreSASHA2POfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA2POfflineSepMeanandStdLst[1].extend(meanandstdT1[1])

                        PreSASHA3PT1MapOnLine = Variable(
                            torch.FloatTensor(PreSASHA3PT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PreSASHA3PT1MapOnLine = PreSASHA3PT1MapOnLine.reshape((xs[0], 1, xs[3], xs[4]))

                        meanandstdT1 = mean_std_ROI(PreSASHA3PT1MapOnLine*TimeScaling, MyoMaskPreSASHA)
                        PreSASHA3POfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA3POfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreSASHA3PT1MapOnLine*TimeScaling, BPMaskPreSASHA)
                        PreSASHA3POfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA3POfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PreSASHA3PT1MapOnLine*TimeScaling, SepMaskPreSASHA)
                        PreSASHA3POfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PreSASHA3POfflineSepMeanandStdLst[1].extend(meanandstdT1[1])

                    except Exception as e:
                        print(e)

                    try:
                        PostSASHA2PT1MapOnLine = Variable(
                            torch.FloatTensor(PostSASHA2PT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PostSASHA2PT1MapOnLine = PostSASHA2PT1MapOnLine.reshape((xs[0], 1, xs[3], xs[4]))

                        MyoMaskPostSASHA = PostSASHAMyoMask[tst_lst[idx:idx + bs]]
                        BPMaskPostSASHA = PostSASHABPMask[tst_lst[idx:idx + bs]]
                        SepMaskPostSASHA = PostSASHASepMask[tst_lst[idx:idx + bs]]

                        meanandstdT1 = mean_std_ROI(PostSASHA2PT1MapOnLine * TimeScaling, MyoMaskPostSASHA)
                        PostSASHA2POfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA2POfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PostSASHA2PT1MapOnLine * TimeScaling, BPMaskPostSASHA)
                        PostSASHA2POfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA2POfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PostSASHA2PT1MapOnLine * TimeScaling, SepMaskPostSASHA)
                        PostSASHA2POfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA2POfflineSepMeanandStdLst[1].extend(meanandstdT1[1])

                        PostSASHA3PT1MapOnLine = Variable(
                            torch.FloatTensor(PostSASHA3PT1MapOffLineT1[tst_lst[idx:idx + bs]])).to(
                            'cuda:0')
                        PostSASHA3PT1MapOnLine = PostSASHA3PT1MapOnLine.reshape((xs[0], 1, xs[3], xs[4]))

                        meanandstdT1 = mean_std_ROI(PostSASHA3PT1MapOnLine * TimeScaling, MyoMaskPostSASHA)
                        PostSASHA3POfflineMyoMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA3POfflineMyoMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PostSASHA3PT1MapOnLine * TimeScaling, BPMaskPostSASHA)
                        PostSASHA3POfflineBPMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA3POfflineBPMeanandStdLst[1].extend(meanandstdT1[1])

                        meanandstdT1 = mean_std_ROI(PostSASHA3PT1MapOnLine * TimeScaling, SepMaskPostSASHA)
                        PostSASHA3POfflineSepMeanandStdLst[0].extend(meanandstdT1[0])
                        PostSASHA3POfflineSepMeanandStdLst[1].extend(meanandstdT1[1])


                    except Exception as e:
                        print(e)


                    # for ix in range(0, bs):
                    #     ij = tst_lst[idx + ix]
                    #     preSubjectsLst.append(subjectsLists[ij])
                    #
                    # PreSubjectsLists.extend(preSubjectsLst)


                    #save pre-Contrast
                    try:
                        n_slices = PreMOLLI5HBsPredT1.shape[0]
                        for sl in range(0, n_slices):

                            subjectName =preSubjectsLst[sl]

                            fig, axs = plt.subplots(1, 5)
                            axs[0].set_title('LL4')
                            TxMap = Pre5HBsPredT1[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            axs[0].imshow(TxMap, cmap='jet', vmin=0, vmax=2500)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[0].axis('off')
                            #save mat file
                            saveArrayToMat(TxMap, 'TxMap', 'Pre5HBsPredT1_'+subjectName, params.TestingResults_dir)

                            axs[1].set_title('MOLLI53-4HBs')
                            TxMap = PreMOLLI5HBsPredT1[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            pos = axs[1].imshow(TxMap, cmap='jet', vmin=0, vmax=2500)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[1].axis('off')
                            # fig.colorbar(pos, ax=axs[0])
                            # cbar = fig.colorbar(pos, ax=axs[0], extend='both')
                            # cbar.minorticks_on()
                            # divider = make_axes_locatable(axs[1])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            #save mat file
                            saveArrayToMat(TxMap, 'TxMap', 'PreMOLLI5HBsPredT1_'+ subjectName, params.TestingResults_dir)


                            axs[2].set_title('MOLLI53')
                            TxMap = PreMOLLIT1MapOFFLine[sl,0,:,:].cpu().data.numpy()*TimeScaling
                            pos = axs[2].imshow(TxMap, cmap='jet', vmin=0, vmax=2500)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[2].axis('off')
                            # divider = make_axes_locatable(axs[2])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PreMOLLIT1MapOffLine_' + subjectName, params.TestingResults_dir)

                            axs[3].set_title('SASHA-2Para')

                            TxMap = PreSASHA2PT1MapOnLine[sl,0,:,:].cpu().data.numpy()*TimeScaling
                            pos = axs[3].imshow(TxMap, cmap='jet', vmin=0, vmax=2500)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[3].axis('off')
                            # divider = make_axes_locatable(axs[3])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PreSASHA2ParaT1MapOffLine_' + subjectName, params.TestingResults_dir)


                            axs[4].set_title('SASHA-3Para')

                            TxMap = PreSASHA3PT1MapOnLine[sl,0,:,:].cpu().data.numpy()*TimeScaling
                            pos = axs[4].imshow(TxMap, cmap='jet', vmin=0, vmax=2500)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[4].axis('off')
                            divider = make_axes_locatable(axs[4])
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PreSASHA3ParaT1MapOffLine_' + subjectName, params.TestingResults_dir)


                            # try:
                            # #save SASHA
                            #     TxMap = PreSASHA2PT1MapOnLine[sl,0,:,:].cpu().data.numpy()*TimeScaling
                            #     saveArrayToMat(TxMap, 'TxMap', 'PreSASHA2PT1Map_' + subjectName, params.TestingResults_dir)
                            #
                            #     TxMap = PreSASHA3PT1MapOnLine[sl,0,:,:].cpu().data.numpy()*TimeScaling
                            #     saveArrayToMat(TxMap, 'TxMap', 'PreSASHA3PT1Map_' + subjectName, params.TestingResults_dir)
                            #     # fig.savefig(params.validation_dir + tst_sliceID[sl] + '.png')
                            #
                            # except Exception as e:
                            #     print(e)

                            fig.savefig(params.TestingResults_dir  + subjectName+ '_Pre.png')
                            plt.close(fig)
                    except Exception as e:
                        traceback.print_exc()
                        # continue

                    # save post-Contrast
                    try:
                        n_slices = PreMOLLI5HBsPredT1.shape[0]
                        for sl in range(0, n_slices):

                            subjectName = preSubjectsLst[sl]

                            fig, axs = plt.subplots(1, 5)
                            axs[0].set_title('LL4')
                            TxMap = Post5HBsPredT1[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            axs[0].imshow(TxMap, cmap='jet', vmin=0, vmax=1200)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[0].axis('off')
                            # save mat file
                            saveArrayToMat(TxMap, 'TxMap', 'Post5HBsPredT1_' + subjectName, params.TestingResults_dir)

                            axs[1].set_title('MOLLI432-4HBs')
                            TxMap = PostMOLLI5HBsPredT1[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            pos = axs[1].imshow(TxMap, cmap='jet', vmin=0, vmax=1200)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[1].axis('off')
                            # fig.colorbar(pos, ax=axs[0])
                            # cbar = fig.colorbar(pos, ax=axs[0], extend='both')
                            # cbar.minorticks_on()
                            # divider = make_axes_locatable(axs[1])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            # save mat file
                            saveArrayToMat(TxMap, 'TxMap', 'PostMOLLI5HBsPredT1_' + subjectName,
                                           params.TestingResults_dir)

                            axs[2].set_title('MOLLI53')
                            TxMap = PostMOLLIT1MapOFFLine[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            pos = axs[2].imshow(TxMap, cmap='jet', vmin=0, vmax=1200)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[2].axis('off')
                            # divider = make_axes_locatable(axs[2])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PostMOLLIT1MapOffLine_' + subjectName,
                                           params.TestingResults_dir)

                            axs[3].set_title('SASHA-2Para')

                            TxMap = PostSASHA2PT1MapOnLine[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            pos = axs[3].imshow(TxMap, cmap='jet', vmin=0, vmax=1200)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[3].axis('off')
                            # divider = make_axes_locatable(axs[3])
                            # cax = divider.append_axes("right", size="5%", pad=0.05)
                            # plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PostSASHA2ParaT1MapOffLine_' + subjectName,
                                           params.TestingResults_dir)

                            axs[4].set_title('SASHA-3Para')

                            TxMap = PostSASHA3PT1MapOnLine[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            pos = axs[4].imshow(TxMap, cmap='jet', vmin=0, vmax=1200)
                            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
                            axs[4].axis('off')
                            divider = make_axes_locatable(axs[4])
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(pos, cax=cax)
                            saveArrayToMat(TxMap, 'TxMap', 'PostSASHA3ParaT1MapOffLine_' + subjectName,
                                           params.TestingResults_dir)

                            # try:
                            #     # save SASHA
                            #     TxMap = PreSASHA2PT1MapOnLine[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            #     saveArrayToMat(TxMap, 'TxMap', 'PreSASHA2PT1Map_' + subjectName,
                            #                    params.TestingResults_dir)
                            #
                            #     TxMap = PreSASHA3PT1MapOnLine[sl, 0, :, :].cpu().data.numpy() * TimeScaling
                            #     saveArrayToMat(TxMap, 'TxMap', 'PreSASHA3PT1Map_' + subjectName,
                            #                    params.TestingResults_dir)
                            #     # fig.savefig(params.validation_dir + tst_sliceID[sl] + '.png')
                            #
                            # except Exception as e:
                            #     print(e)

                            fig.savefig(params.TestingResults_dir + subjectName + '_Post.png')
                            plt.close(fig)
                    except Exception as e:
                        traceback.print_exc()
                        # continue

                except Exception as e:
                    traceback.print_exc()
                    # continue
            #export all resutls as one mat file
            allResults['Pre5HBsPredMyoMeanandStd'] = np.array(Pre5HBsPredMyoMeanandStdLst).transpose()
            allResults['Pre5HBsPredBPMeanandStd'] = np.array(Pre5HBsPredBPMeanandStdLst).transpose()
            allResults['Pre5HBsPredSepMeanandStd'] = np.array(Pre5HBsPredSepMeanandStdLst).transpose()

            allResults['Post5HBsPredMyoMeanandStd'] = np.array(Post5HBsPredMyoMeanandStdLst).transpose()
            allResults['Post5HBsPredBPMeanandStd'] = np.array(Post5HBsPredBPMeanandStdLst).transpose()
            allResults['Post5HBsPredSepMeanandStd'] = np.array(Post5HBsPredSepMeanandStdLst).transpose()

            allResults['PreMOLLIPredMyoMeanandStd'] = np.array(PreMOLLIPredMyoMeanandStdLst).transpose()
            allResults['PreMOLLIPredBPMeanandStd'] = np.array(PreMOLLIPredBPMeanandStdLst).transpose()
            allResults['PreMOLLIPredSepMeanandStd'] = np.array(PreMOLLIPredSepMeanandStdLst).transpose()

            allResults['PreMOLLIOfflineMyoMeanandStd'] = np.array(PreMOLLIOfflineMyoMeanandStdLst).transpose()
            allResults['PreMOLLIOfflineBPMeanandStd'] = np.array(PreMOLLIOfflineBPMeanandStdLst).transpose()
            allResults['PreMOLLIOfflineSepMeanandStd'] = np.array(PreMOLLIOfflineSepMeanandStdLst).transpose()

            allResults['PostMOLLIPredMyoMeanandStd'] = np.array(PostMOLLIPredMyoMeanandStdLst).transpose()
            allResults['PostMOLLIPredBPMeanandStd'] = np.array(PostMOLLIPredBPMeanandStdLst).transpose()
            allResults['PostMOLLIPredSepMeanandStd'] = np.array(PostMOLLIPredSepMeanandStdLst).transpose()

            allResults['PostMOLLIOfflineMyoMeanandStd'] = np.array(PostMOLLIOfflineMyoMeanandStdLst).transpose()
            allResults['PostMOLLIOfflineBPMeanandStd'] = np.array(PostMOLLIOfflineBPMeanandStdLst).transpose()
            allResults['PostMOLLIOfflineSepMeanandStd'] = np.array(PostMOLLIOfflineSepMeanandStdLst).transpose()

            allResults['PreSASHA2POfflineMyoMeanandStd'] = np.array(PreSASHA2POfflineMyoMeanandStdLst).transpose()
            allResults['PreSASHA2POfflineBPMeanandStd'] = np.array(PreSASHA2POfflineBPMeanandStdLst).transpose()
            allResults['PreSASHA2POfflineSepMeanandStd'] = np.array(PreSASHA2POfflineSepMeanandStdLst).transpose()

            allResults['PreSASHA3POfflineMyoMeanandStd'] = np.array(PreSASHA3POfflineMyoMeanandStdLst).transpose()
            allResults['PreSASHA3POfflineBPMeanandStd'] = np.array(PreSASHA3POfflineBPMeanandStdLst).transpose()
            allResults['PreSASHA3POfflineSepMeanandStd'] = np.array(PreSASHA3POfflineSepMeanandStdLst).transpose()

            allResults['PostSASHA2POfflineMyoMeanandStd'] = np.array(PostSASHA2POfflineMyoMeanandStdLst).transpose()
            allResults['PostSASHA2POfflineBPMeanandStd'] = np.array(PostSASHA2POfflineBPMeanandStdLst).transpose()
            allResults['PostSASHA2POfflineSepMeanandStd'] = np.array(PostSASHA2POfflineSepMeanandStdLst).transpose()

            allResults['PostSASHA3POfflineMyoMeanandStd'] = np.array(PostSASHA3POfflineMyoMeanandStdLst).transpose()
            allResults['PostSASHA3POfflineBPMeanandStd'] = np.array(PostSASHA3POfflineBPMeanandStdLst).transpose()
            allResults['PostSASHA3POfflineSepMeanandStd'] = np.array(PostSASHA3POfflineSepMeanandStdLst).transpose()


            allResults['PreSubjectsLists'] = PreSubjectsLists
            allResults['PostSubjectsLists'] = PostSubjectsLists

            saveArrayToMat(allResults, 'allResults',
                           'MeanandStdT1ofMyoandBP',
                           params.TestingResults_dir)
            #plot bland-altman  here
            #ax = pg.plot_blandaltman(np.array(allMyoBPPixelsPredict)[0], np.array(allMyoBPPixelsPredict)[0])
            #pyCompare.blandAltman(np.array(meanT1Pre), np.array(meanT1Ref))

    except Exception as e:
            traceback.print_exc()
            # print(e)
            #continue
    # writer.close()


ssimCriterion = SSIM()
mseCriterion = Loss.MSELoss()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Model Saved!')
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params.args.lr * (0.1 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mean_T1(x, mask):
    meant1 = list()
    for i in range(0, x.shape[0]):
        xs = x[i,]
        myo_T1 = xs[np.nonzero(mask[i,].cpu().data.numpy())].cpu().data.numpy()
        myo_T1 = myo_T1[myo_T1 > 1500]
        myo_T1 = myo_T1[myo_T1 < 2200]
        # meant1.append(myo_T1.std())
        meant1.append(myo_T1.mean())
    return meant1

def mean_std_ROI(x, mask):
    meanstdValsArr = [ [0 for i in range(x.shape[0])] for i in range(2)]
    for i in range(0, x.shape[0]):
        xs = x[i,]
        roiVals = xs[np.nonzero(mask[i,])].cpu().data.numpy()
        meanstdValsArr[0][i] = roiVals.mean()
        meanstdValsArr[1][i] = roiVals.std()
    return meanstdValsArr

def get_allPixels(x, mask):
    allPixles = list()
    for i in range(0, x.shape[0]):
        xs = x[i,]
        allpixlestmp = xs[np.nonzero(mask[i,].cpu().data.numpy())].cpu().data.numpy()
        # myo_T1 = myo_T1[myo_T1 > 1500]
        # myo_T1 = myo_T1[myo_T1 < 2200]
        # # meant1.append(myo_T1.std())
        allPixles.append(allpixlestmp)
    return allPixles

if __name__ == '__main__':

    try:
        if params.Training_Only:
            train(MyoMapNet)
        else:
            Testing(MyoMapNet)
    except KeyboardInterrupt:
        print('Interrupted')
        torch.save(MyoMapNet.state_dict(), 'MODEL_INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
