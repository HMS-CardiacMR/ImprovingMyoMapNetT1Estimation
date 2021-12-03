'''
Created on May 17, 2018
@author: helrewaidy
'''
# models

import argparse

from pycparser.c_ast import Switch
import torch
import numpy as np
import os

########################## Initializations ########################################
model_names = 'recoNet_Model1'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--cpu', '-c', action='store_true',
                    help='Do not use the cuda version of the net',
                    default=False)
parser.add_argument('--viz', '-v', action='store_true',
                    help='Visualize the images as they are processed',
                    default=False)
parser.add_argument('--no-save', '-n', action='store_false',
                    help='Do not save the output masks',
                    default=False)
parser.add_argument('--model', '-m', default='MODEL_EPOCH417.pth',
                    metavar='FILE',
                    help='Specify the file in which is stored the model'
                         " (default : 'MODEL.pth')")
###################################################################

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class Parameters():
    def __init__(self):
        super(Parameters, self).__init__()

        ## Hardware/GPU parameters =================================================
        self.Op_Node = 'spider'  # 'alpha_V12' # 'myPC', 'O2', 'spider'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tbVisualize = False
        self.tbVisualize_kernels = False
        self.tbVisualize_featuremaps = False
        self.multi_GPU = True

        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.device_ids = [0]
        elif self.Op_Node in ['spider', 'O2']:
            self.device_ids = range(0, torch.cuda.device_count())

        if self.Op_Node in ['spider', 'O2', 'alpha_V12']:
            self.data_loders_num_workers = 40
        else:
            self.data_loders_num_workers = 4

        ## Network/Model parameters =================================================
        self.network_type = '2D'
        self.num_slices_3D = 7
        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.batch_size = 2
        elif self.Op_Node in ['spider', 'O2']:
            self.batch_size = 4 # * len(self.device_ids) // 8 // (self.num_slices_3D if self.network_type == '3D' else 1)

        print('-- # GPUs: ', len(self.device_ids))
        print('-- batch_size: ', self.batch_size)
        self.args = parser.parse_args()

        self.activation_func = 'CReLU'  # 'CReLU' 'CLeakyeak' # 'modReLU' 'KAF2D' 'ZReLU'
        #self.args.lr = 1e-8 #0.0001
        #self.args.lr = 1e-2  # 0.01  #Changed by Rui Guo
        # self.args.lr = 1e-6  # for '4HBsPrelowLR'
        self.args.lr = 1e-3 # for all model
        #self.args.lr = 1e-2
        #self.args.lr = 5e-4  # 0.001  #Changed by Rui Guo
        self.dropout_ratio = 0.0
        self.epochs = 3000     #simmyo1000,simblood 1000, in-vivo 200
        self.training_percent = 0.8
        self.nIterations = 1
        self.magnitude_only = False
        self.batch_size = 20

        self.Training_Only = False
        self.Validation_Only = False  # option for training or validation
        self.Testing_Only = True
        # option for training or validation
        self.training_with_Sim = True
        self.training_with_Phantom = False
        self.training_with_Invivo = False


        self.validation_with_Phantom = True
        # self.Evaluation = Falsec
        #self.Testing = False

        self.NetName = 'BsNet'  #trained for native T1
        # self.NetName = '4HBsPrermNoiseMask'   #Note, this is used for study the performance of different dataset
        #self.NetName = '4HBsPrelowLR'
        #self.NetName = '4HBsPre33Patch'
        self.inputLen = 8
        self.inputLen = 4

        ##options for Model
        #self.MODEL = 0 # Original U-net implementation
        #self.MODEL = 1 # Shallow U-net implementation with combination of magnitude and phase
        #self.MODEL = 2 # #The OLD working Real and Imaginary network (Residual Network with one global connection)
        #self.MODEL = 3 # Complex Shallow U-net
        #self.MODEL = 3.1 # Complex stacked convolution layers
        #self.MODEL = 3.2 # Complex Shallow U-net with different kernel configuration
        #self.MODEL = 3.3 # Complex 3D U-net with multi GPU implemntation to fit whole 3D volume
        #self.MODEL = 3.4 # Complex 3D U-net with multi GPU implemntation to fit whole 3D volume
        #self.MODEL = 4 # Complex Shallow U-Net with residual connection
        #self.MODEL = 5 # Complex Shallow U-Net with residual connection with 32 multi coil output
        #self.MODEL = 6 # Complex fully connected layer
        #self.MODEL = 7 # Real shallow U-net layer [double size]
        #self.MODEL = 8 # complex conv network that maps k-space to image domain
        #self.MODEL = 9  # Complex Network takes neighborhood matrix input and image domain output
        self.MODEL = 10
        #########
        if self.MODEL in [2, 3, 3.1, 3.2, 3.3, 3.4, 4, 5, 6, 8, 9]:
            self.complex_net = True
        else:
            self.complex_net = False

        ## Dataset and paths =================================================
        self.g_methods = ['grid_kernels', 'neighbours_matrix', 'pyNUFFT', 'BART', 'python_interp']
        self.gridding_method = 'MOLLI_MIRT_NUFFT'

        self.gd_methods = ['RING', 'AC-ADDAPTIVE', 'NONE']
        self.gradient_delays_method = ''  # self.gd_methods[0]
        self.rot_angle = True

        self.k_neighbors = 20

        self.ds_total_num_slices = 0
        self.patients = []
        self.num_phases = 5
        self.radial_cine = False

        ## for acceleration of MOLLI projects
        self.AccPreMOLLI = True
        self.AccPostMOLLI = False

        self.n_spokes = 40  # 16 #20 #33
        self.Rate = np.round(198 / self.n_spokes) ##if self.radial_cine else 3
        self.input_slices = list()
        self.num_slices_per_patient = list()
        self.groundTruth_slices = list()
        self.training_patients_index = list()
        self.us_rates = list()
        self.saveVolumeData = False
        self.multiCoilInput = True
        self.coilCombinedInputTV = True
        self.moving_window_size = 5

        if self.network_type == '2D':
            self.img_size = [416, 416]
        else:
            self.img_size = [416, 416, self.moving_window_size]  # [50, 50, 20]  # 64, 256, 320

        if self.multiCoilInput:
            self.n_channels = 1
        else:
            self.n_channels = 1

        self.cropped_dataset64 = False
        if self.cropped_dataset64:
            crop_txt = '_cropped64'
        else:
            crop_txt = ''
        self.trialNum = '_T1Cal_MAE_alldata' #best results '_T1_5016' and 5031, and _T1_5041_MOLLI
        self.arch_name = 'Model_Net_' + str(self.MODEL) + '_Trial_NetName_'+ self.NetName +'_' + self.trialNum

        if self.Op_Node == 'alpha_V12':
            if self.coilCombinedInputTV:
                self.dir = {'/mnt/D/Image Reconstruction-Hossam/NoDataset/',
                            '/mnt/C/Hossam/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                            }
            else:
                self.dir = {'/mnt/D/Image Reconstruction-Hossam/Dataset/ReconData_cmplxDL/',
                            '/mnt/C/Hossam/ReconData_cmplxDL/'
                            }
            self.model_save_dir = '/mnt/C/Hossam/RecoNet-Model/' + self.arch_name + '/'
            self.net_save_dir = '/mnt/D/Image Reconstruction-Hossam/MatData/'
            self.tensorboard_dir = '/mnt/C/Hossam/RecoNet-Model/' + self.arch_name + '_tensorboard/'


        elif self.Op_Node == 'myPC':
            if self.coilCombinedInputTV:
                self.dir = {'/media/helrewaidy/F/Image Reconstruction/ReconData_coilCombTVDL/Rate_' + str(
                    self.Rate) + crop_txt + '/',
                            '/mnt/D/BIDMC Workspace/Image Reconstruction/ReconData_coilCombTVDL/Rate_' + str(
                                self.Rate) + '/'
                            }
            else:
                self.dir = {'/mnt/C/Image Reconstruction/ReconData_cmplxDL/',
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/',
                            '/mnt/D/BIDMC Workspace/Image Reconstruction/ReconData_cmplxDL/'
                            }
            self.model_save_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/RecoNet-Model/' + self.arch_name + '/'
            self.net_save_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/MatData/'
            self.tensorboard_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/RecoNet-Model/' + self.arch_name + '_tensorboard/'


        elif self.Op_Node == 'O2':
            if self.coilCombinedInputTV:
                self.dir = {'/n/scratch2/hae1/ReconData/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                            }
            else:
                self.dir = {'/n/scratch2/hae1/ReconData/ReconData_cmplxDL/'
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                            }
            self.model_save_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/RecoNet_Model/' + self.arch_name + '/'
            self.net_save_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/MatData/'
            self.tensorboard_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/RecoNet_Model/' + self.arch_name + '_tensorboard/'



        elif self.Op_Node == 'spider':

            if self.radial_cine:
                self.dir = ['/data1/helrewaidy/cine_recon/ICE_recon_dat_files/ice_dat_files/'
                            ]
                # ['/data2/helrewaidy/cine_recon/ICE_recon_dat_files/ice_dat_files/'
                #  ]
                self.model_save_dir = '/data2/helrewaidy/cine_recon/models/' + self.arch_name + '/'
                self.net_save_dir = '/data2/helrewaidy/t1mapping_recon/matlab_workspace/' + self.arch_name + '/'
                self.tensorboard_dir = '/data2/helrewaidy/cine_recon/models/' + self.arch_name + '_tensorboard/'

            elif self.AccPreMOLLI:
                self.model_save_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Models/' + self.arch_name + '/'
                self.net_save_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/NetSaveDir/' + self.arch_name + '/'
                self.tensorboard_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Models/' + self.arch_name + '_tensorboard/'
                self.tensorboard_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Models/' + self.arch_name + '_tensorboardLoss/'

                self.validation_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Models/' + self.arch_name + '_validation/PreContrast/'
                self.validation_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Models/' + self.arch_name + '_validation/Loss/'

                # self.TestingResults_dir = '/data2/rguo/Projects/BsNetMOLLI/Models/' + self.arch_name + '_testing/Retrospective_allPreContrast'+ '_'+self.arch_name+'//'
                #####################################Training###############################################################################
                # self.TrainSimFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Numerical_simulation/AllSimSignals_T2UD_25-Sep-2021.mat'
                self.TrainSimFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Numerical_simulation/AllSimSignals_T2UD_ReduceB1Off_05-Oct-2021.mat'
                # self.TrainPhantomFile ='/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Phantom/20210904/allPhantomT1map_208_188_Training.mat'
                # self.TrainPhantomFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Phantom/20210904/AllT1_100_2500_T2_20_250_B1_3.0009_20.7691_HR_30_130_SNR_80_120_OFF_204.0654_201.7436SliceProfils_1_ExpT2_INV1_4000000.mat'
                self.TrainPhantomFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Phantom/20211016/allPhantomT1data_Training.mat'
                #####################################Validation###############################################################################
                # self.ValidationPhantomFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Phantom/20210904/allPhantomT1map_208_188_Validation.mat'
                self.ValidationPhantomFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Phantom/20211016/allPhantomT1data_Validation.mat'


                self.preBloodSimFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Preconstrast_numerical_simulation/precontrastBlood_s1500000.mat'

                self.postMyoSimFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Postconstrast_numerical_simulation/postcontrastMyo_s1500000.mat'
                self.postBloodSimFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Postconstrast_numerical_simulation/postcontrastBlood_s1500000.mat'


                self.TrainPrePostInvivo_5HBs = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/allInvivoPrePostT1map_160_160_CircleMask.mat'
                self.TrainingPreInvivoFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Precontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160_Circlemask.mat'
                self.TrainingPostInvivoFile = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Postcontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160_Circlemask.mat'



                #####################Testing###############################
                self.TestingPhantom = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/Phantom_Date_20210810/Phantom_MAT/allPhT1map_160_160_Testing.mat'

                self.TestingSimPrecontrast = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/NumericalSimulation/SimData/allSim_pre_T1map_50_50_SL5_Testing.mat'
                self.TestingSimPostcontrast = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/NumericalSimulation/SimData/allSim_post_T1map_50_50_SL5_Testing.mat'
                self.TestingResults_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/ResultsNet/' + self.arch_name + '_testing/Invivo_T1_500_1200_T2_20_60_B1_10_10_HR_30_120_SNR_180_180_OFF_0_0_Os_Spider_SNR180_Inv096_s500000/'

                self.TestingInvivoProspective = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/Prospeictive_Invivo_withSASHA/MAT/allInvivoT1map_208_188_Testing.mat'

                self.TestingResults_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/NetOutput/Phantom_20210810/'
                # self.TestingResults_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/NetOutput/Simulation/Phantom/'
                # self.TestingResults_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/NetOutput/Simulation/Invivo/'
                self.TestingResults_dir = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/NetOutput/Invio_prospective/'
            else:
                if self.coilCombinedInputTV:
                    self.dir = ['/data2/helrewaidy/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                                # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                                ]
                else:
                    self.dir = {'/n/scratch2/hae1/ReconData/ReconData_cmplxDL/'
                                # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                                }
                self.model_save_dir = '/data2/helrewaidy/Models/ReconNet_Model/' + self.arch_name + '/'
                self.net_save_dir = '/data2/helrewaidy/Models/MatData/'
                self.tensorboard_dir = '/data2/helrewaidy/Models/ReconNet_Model/' + self.arch_name + '_tensorboard/'

        self.args.model = self.model_save_dir + 'MODEL_EPOCH.pth'
