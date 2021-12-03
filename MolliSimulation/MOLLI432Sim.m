%[oSig, oTinv]= MOLLI432Sim(HRs, iT1, iT2, invEff, ioffFre, iB1Scale)
% Simulate Molli4(1)3(1)2 T1 mapping sequence
%
%[oSig, oTinv] = MOLLI432Sim(HRs, iT1, iT2)
%[oSig, oTinv] = MOLLI432Sim(HRs, iT1, iT2, invEff)
%[oSig, oTinv] = MOLLI432Sim(HRs, iT1, iT2, invEff, ioffFre)
%[oSig, oTinv] = MOLLI432Sim(HRs, iT1, iT2, invEff, ioffFre, iB1Scale)
%
%Input
%   HRs=    simulated heart rate
%   iT1=    simulated T1
%   iT2=    simulated T2
%   invEff= Effiency of inversion pulse (Default is 1).  it is ranging from 0 to 1
%   ioffFre=  simulated off resonance (Hz), default is 0. At 3T, the simulated off-resonance should 
%be ranged from -100 Hz to 100 Hz. (Kellman P, Herzka DA, Arai AE, Hansen MS. Influence of Off-resonance 
%in myocardial T1-mapping using SSFP based MOLLI method. J Cardiovasc Magn Reson. 2013; 15:63.)
%   iB1Scale = scale factor for B1 field, default is 1.
%
%Output
%   oflg = 1:success  0:failure
%   oSig = Magnetizions [Mx, My, Mz] of K-sapce center line of eight images
%   oTinv = Inversion-recovery time of eight images 
% by Rui Guo
function [oflg, oSig, oTinv] = MOLLI432Sim(HRs, iT1, iT2, invEff, ioffFre, iB1Scale)



if nargin < 4 || isempty(invEff)
    invEff = 1;     %100% in
end

if nargin < 5 || isempty(ioffFre)
    ioffFre = 0;     %frequeence (Hz) of off-resonance, 
end 

if nargin < 6 || isempty(iB1Scale)
    iB1Scale = 1;     %B1 scale factor
end 

Meq = 1;
%parameters related to bSSFP
T1 = iT1;
T2 = iT2;
offFre = ioffFre;
B1scale = iB1Scale;
TR = 2.5; %ms
KspaceLens = 80+5; %including 5 ramp-up pulses
ACQWindow = KspaceLens*TR; %ms
durbeforeKcenter = (5+40)*TR;

%parameters related to MOLLI53
Tinv1 = 120;
Tinv2 = 200;
Tinv3 = 280;

cardiacCycleNum = 11;


if(length(HRs)==1)
    HRArr = ones(cardiacCycleNum,1)*HRs;
else
    HRArr = HRs(1:cardiacCycleNum);
    HRArr = HRArr(:);
end
cardiacCycleArrs = fix(60./HRArr*1000); %ms

%calculate the middle diastolic delay
triggerDealyArrs = zeros(cardiacCycleNum,1);

K1 = 0.380;
K2 = 0.07;

for ix =1:cardiacCycleNum
    RRInterval = cardiacCycleArrs(ix);
    Tsys = K1*log10(10*(RRInterval/1000+K2))*1000;
    Tdiastole = RRInterval - Tsys;
    TmidDias = Tsys +Tdiastole/2;
    
    if((RRInterval- TmidDias - ACQWindow)>0 )
        triggerDealyArrs(ix) = fix(TmidDias);
    else
        triggerDealyArrs(ix) = fix(RRInterval- ACQWindow-10);
        if( triggerDealyArrs(ix)<0)
             oflg= 0;
             oSig = [];
             oTinv = [];
            return;
        end
    end
end
%1(inv+ACQ)-2(ACQ)-3(ACQ)-4(ACQ)-5(Dummy)-6(inv+ACQ)-7(ACQ)-8(ACQ)-9(Dummy)-10(inv+ACQ)-11(ACQ)
%set the Tinv timing
delayBwnInvAcq1 = Tinv1 - durbeforeKcenter;
if(delayBwnInvAcq1>triggerDealyArrs(1))
    delayBwnInvAcq1 = triggerDealyArrs(1);
    Tinv1 = delayBwnInvAcq1+durbeforeKcenter;
end

delayBwnInvAcq2 = Tinv2 - durbeforeKcenter;
if(delayBwnInvAcq2>triggerDealyArrs(6))
    delayBwnInvAcq2 = triggerDealyArrs(6);
    Tinv2 = delayBwnInvAcq2+durbeforeKcenter;
end

delayBwnInvAcq3 = Tinv3 - durbeforeKcenter;
if(delayBwnInvAcq3>triggerDealyArrs(10))
    delayBwnInvAcq3 = triggerDealyArrs(10);
    Tinv3 = delayBwnInvAcq3+durbeforeKcenter;
end

%Perform MOLLI53 sequence
TinvArrs = zeros(9,1);
SigsaArr = zeros(9,3);

%%
%-------------------------------Simulation of IR sequence --------------------------%
%       |R---------------------------------------------------------------------------------R| 
%       |<-----------------------------cardiac cycle -------------------------------------->|
%       |--------------------------------Inv-------------ACQ(bSSFP)-------------------------|
%       ||<-----------------------trigger delay-------->|          |<---Dealy after ACQ---->|
%       |                                 |<--------Tinv----->|                             |
%       |MzbeforeRR            MzbeforeINV   MzbeforeACQ|          |MzAfterACQ  MzafterDummy|
%       


%1st RR
MzbeforeInv = Meq;
cardiacCycle = cardiacCycleArrs(1,1);
triggerDealy = triggerDealyArrs(1,1);
MzafterInv = -1*MzbeforeInv*invEff;
MzbeforeAcq = MzafterInv*exp(-delayBwnInvAcq1/T1)+Meq*(1-exp(-delayBwnInvAcq1/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(1,1) = Tinv1;
SigsaArr(1,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%2nd RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(2,1);
triggerDealy = triggerDealyArrs(2,1);
TDbeforeAcq = triggerDealyArrs(2,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(2,1) = TinvArrs(1,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(2,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%3rd RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(3,1);
triggerDealy = triggerDealyArrs(3,1);
TDbeforeAcq = triggerDealyArrs(3,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(3,1) = TinvArrs(2,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(3,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%4th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(4,1);
triggerDealy = triggerDealyArrs(4,1);
TDbeforeAcq = triggerDealyArrs(4,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(4,1) = TinvArrs(3,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(4,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%5th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(5,1);
TDafterAcq = cardiacCycle;
MzafterDummy =MzbeforeR*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%6th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(6,1);
triggerDealy = triggerDealyArrs(6,1);
TDbeforeInv = triggerDealy -delayBwnInvAcq2;
MzbeforeInv = MzbeforeR*exp(-TDbeforeInv/T1)+Meq*(1-exp(-TDbeforeInv/T1));
MzafterInv = -MzbeforeInv*invEff;
TDbeforeAcq = delayBwnInvAcq2;
MzbeforeAcq = MzafterInv*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(5,1) = Tinv2;
SigsaArr(5,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%7th
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(7,1);
triggerDealy = triggerDealyArrs(7,1);
TDbeforeAcq = triggerDealyArrs(7,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(6,1) = TinvArrs(5,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(6,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%8th
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(8,1);
triggerDealy = triggerDealyArrs(8,1);
TDbeforeAcq = triggerDealyArrs(8,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(7,1) = TinvArrs(6,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(7,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));

%9th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(9,1);
TDafterAcq = cardiacCycle;
MzafterDummy =MzbeforeR*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%10th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(10,1);
triggerDealy = triggerDealyArrs(10,1);
TDbeforeInv = triggerDealy -delayBwnInvAcq3;
MzbeforeInv = MzbeforeR*exp(-TDbeforeInv/T1)+Meq*(1-exp(-TDbeforeInv/T1));
MzafterInv = -MzbeforeInv*invEff;
TDbeforeAcq = delayBwnInvAcq3;
MzbeforeAcq = MzafterInv*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(8,1) = Tinv3;
SigsaArr(8,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));


%11th RR
MzbeforeR = MzafterDummy;
cardiacCycle = cardiacCycleArrs(11,1);
triggerDealy = triggerDealyArrs(11,1);
TDbeforeAcq = triggerDealyArrs(11,1);
MzbeforeAcq = MzbeforeR*exp(-TDbeforeAcq/T1)+Meq*(1-exp(-TDbeforeAcq/T1));
%ACQ
[simMagnetzaion, KcenterMagnetzaion]=GenerateT1wSignal(T1,T2,MzbeforeAcq,offFre,B1scale);
TinvArrs(9,1) = TinvArrs(8,1)+ACQWindow+TDafterAcq+TDbeforeAcq;
SigsaArr(9,:) = KcenterMagnetzaion;
MzafterAcq = simMagnetzaion(3,end);
TDafterAcq = cardiacCycle - triggerDealy - ACQWindow;
MzafterDummy =MzafterAcq*exp(-TDafterAcq/T1)+Meq*(1-exp(-TDafterAcq/T1));

%--------------------------------------------------------------------------------------------------%
%otuput
oflg = 1;
oSig = SigsaArr;
oTinv = TinvArrs;
end