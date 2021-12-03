%%Test
clear all;
close all;

Tp = 2.56*1E-3; %second
samplingPointsNum = 512;
deltaTp = Tp/(samplingPointsNum-1);
tArrs = 0:deltaTp:Tp;
halfsamplingPts = fix(samplingPointsNum/2);
firstHalfTArrs = tArrs(1:halfsamplingPts);
secondHalTArrs = tArrs(halfsamplingPts+1:end);
B1Max = 1;
xI = 10;
firstHalf_B1_t = B1Max*tanh(4*xI*firstHalfTArrs/Tp);
secondHalf_B1_t = B1Max*tanh(4*xI*(1-secondHalTArrs/Tp));

B1_t = [firstHalf_B1_t secondHalf_B1_t]';
figure, hold on; 
%axis off;
%axis equal;
plot([0:samplingPointsNum-1],B1_t,'*b');
plot([0:samplingPointsNum-1],B1_t,'--r');
hold off;

t = 2.56/2;

phiMax = -

phiMax = 5.469073772;
phiT = phiMax -((2*pi*9500*Tp)/(atan(22)*22))*log(cos(atan(22)*secondHalTArrs/Tp)./cos(atan(22)));