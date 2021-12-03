
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from FCMyoMapNet import UNet
import torch
from torch.autograd import Variable
import numpy as np


#scaling factor for input inverion time and output T1,
TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling
T1sigNum = 4
T1sigAndTi = T1sigNum*2;
# set seed points
seed_num = 888

MyoMapNet = UNet(T1sigAndTi, 1)
MyoMapNet.to(torch.device('cpu'))

modelName = "MODEL_EPOCH945"
#loading trained model
try:
    model = torch.load( 'H:/Projects/BSNetMOLLI/Github/ImprovingMyoMapNetT1Estimation/TrainedModel/' +modelName+'.pth', map_location=torch.device('cpu'))
    MyoMapNet = torch.nn.DataParallel(MyoMapNet)
    MyoMapNet.load_state_dict(model['state_dict'])
    print('Model loaded!')
except Exception as e:
    print('Can not load model!')
    print(e)

print('Start loading demo data')

data = loadmat("H:/Projects/BSNetMOLLI/Github/ImprovingMyoMapNetT1Estimation/Data/Phantom/Testing.mat")

gBS=45
#load prospective pre-contrast testing dataset
try:
    Pre5HBsT1wTIs_in = data['Pre5HBsT1wTIs']
    Pre5HBsT1wTIs_double = Pre5HBsT1wTIs_in.astype(np.double)
    Pre5HBs_tst_t1w_TI = np.abs(Pre5HBsT1wTIs_double[:,:,:,:,:])
except Exception as e:
    print(e)


Pre5HBsDat = Variable(torch.FloatTensor(Pre5HBs_tst_t1w_TI))
xs = Pre5HBsDat.shape
Pre5HBsDat = Pre5HBsDat.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
MyoMapNet.eval()
Pre5HBsPred = MyoMapNet(Pre5HBsDat)
Pre5HBsPredT1 = Pre5HBsPred.reshape((xs[0], 1, xs[3], xs[4]))
plt.imshow(Pre5HBsPredT1[0,0,:,:].data.numpy()*TimeScaling, cmap='jet', vmin=0, vmax=2000)
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.title("Phantom T1 map")
plt.show()