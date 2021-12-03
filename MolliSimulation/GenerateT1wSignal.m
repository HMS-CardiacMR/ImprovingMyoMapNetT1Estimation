%[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(iT1,iT2,iMz,ioffFre,iB1Scale)
%
% generating T1 weighted signal using bSSFP wiht linear K-space filling order. the other parameters are:
% flip angle = 35(°), K-space lines =80, ramp-up RF pulses=5, TR = 2.5 ms
%
%[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(iT1,iT2,iMz)
%[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(iT1,iT2,iMz,ioffFre)
%[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(iT1,iT2,iMz,ioffFre,iB1Scale)
%
%Input:
%   iT1=    simulated T1
%   iT2=    simulated T2
%   iMz=    input Mz
%   ioffFre=     frequence for simulating off-resonance, default is 0
%   iB1Scale=    B1 scale factor, default is 1
%
%Output:
%   simMagnetzaion = [Mx,My, Mz] of each readout line
%   KcenterMagnetzaion = [Mx,My, Mz] of center line

function [simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(iT1,iT2,iMz,ioffFre,iB1Scale,iPhaseNum)

if nargin < 4 || isempty(ioffFre)
    ioffFre = 0;     %frequeence (Hz) of off-resonance, 
end 

if nargin < 5 || isempty(iB1Scale)
    iB1Scale = 1;     %B1 scale factor
end 

if nargin < 6 || isempty(iPhaseNum)
    iPhaseNum = 150;     % 
end 


simT1 = iT1;
simT2 = iT2;
offFre =ioffFre; 
B1scale = iB1Scale;
initMz = iMz;


TRms = 2.47; %ms
GRAPPA_R = 2;
phaseNum = fix(iPhaseNum/GRAPPA_R);
halfPhaseNum = fix(phaseNum/2);
partialFourierFactor = 7/8; 
PostPFPhaseNum = fix(phaseNum*partialFourierFactor);
kSpaceCenterNum = PostPFPhaseNum - halfPhaseNum;
rampUpRF = 5;
KspaceLens = PostPFPhaseNum+rampUpRF; %including 5 ramp-up pulses



nLinesPerACQ = PostPFPhaseNum;
FADeg = 35*B1scale; %degree

%five ramp-up pulse 
rampUpPusle = [FADeg/(rampUpRF*2):FADeg/rampUpRF:(FADeg/(rampUpRF*2)+ FADeg/rampUpRF*(rampUpRF-1))];





nFADeg = [rampUpPusle, ones(1,nLinesPerACQ)*FADeg]/180*pi;
simMagnetzaion = bSSFP(nFADeg,TRms,length(nFADeg),simT1,simT2,initMz, offFre);
KcenterMagnetzaion = [simMagnetzaion(:,kSpaceCenterNum+length(rampUpPusle))]'; 
end




