
%This script is used to generate simualted data for evaluation
%%

clear all
matPath = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Testing/NumericalSimulation/';

%mkdir(matPath);
dugDisp = 0;
%SIEMENS_TanTanh = load('TanTanh\SIEMENSTanTtanh.csv');

%Different T1

Sx = 50;
Sy = 50;
totalSamples = Sx*Sy;
t1RandomIx = randperm(totalSamples);
t2RandomIx = randperm(totalSamples);
offResoanceRandomIx = randperm(totalSamples);
b1RandomIx = randperm(totalSamples);
inversioneffiencyRandomIx = randperm(totalSamples);
hrRandomIx = randperm(totalSamples);
snrRandomIx = randperm(totalSamples);

%for T1
% t1MU = 1500;
% t1SD = 200;
%t1Arrs = normrnd(t1MU,t1SD,totalSamples,1);
myoT1 = 1100:100:1700;
bpT1 = 1700:100:2300;
myoLen = length(myoT1);
bpLen = length(bpT1);

if(~isequal (myoLen,bpLen ))
    disp('Please make sure the length of T1s for myocardium and Blood is same')
    return;
end

%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 45*ones(1, myoLen);
bpT2 = 200*ones(1,bpLen);

%for off-resonance
offResArrs = 0*ones(1,myoLen);
%for B1
b1Arrs =  ones(1,myoLen);
%for inversion pulse
inveArrs = ones(1,myoLen);
%for HR
hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);


%% Different T1

T1wNum = 8;
sliceCount = 1;
OneSecond = 1000; %ms
TimescalingFactor=  1/OneSecond;

PreMOLLIT1wTIs=[];
PreMOLLIT1wTIs=[];
PreMOLLIT1MapOffLine=[];
PreMOLLIT1MapOnLine=[];
SASHA2PT1MapOffLine=[];
SASHA3PT1MapOffLine=[];
PreMOLLIMyoMask=[];
PreMOLLIBPMask=[];
%subjectsLists = [];
sliceNum = 5;
InvDur = 2.56;
% t1Arrs = 1200;
%For myo 
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    subjectsLists(sliceCount,:) = char(['Different_Myo_T1_' num2str(iT1) '____'])
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

%For bp


%For bloodT1
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
     
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    subjectsLists(sliceCount,:) = char(['Different_Bp_T1_' num2str(iT1) '_____'])
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end



%%Different T2

%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 20:10:80;
myoLen = length(myoT2);
bpT2 = 100:20:220;
bpLen = length(bpT2);
if(~isequal (myoLen,bpLen ))
    disp('Please make sure the length of T1s for myocardium and Blood is same')
    return;
end
%for T1
%Myo T1 = 1500;
%blood T1 = 2000;
myoT1 = 1500*ones(1,myoLen);
bpT1 = 2000*ones(1,myoLen);

%for off-resonance
offResArrs = 0*ones(1,myoLen);
%for B1
b1Arrs =  ones(1,myoLen);
%for inversion pulse
inveArrs = ones(1,myoLen);
%for HR
hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);

% t1Arrs = 1200;

%For MyoT2
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
     
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    subjectsLists(sliceCount,:) = char(['Different_Myo_T2_' num2str(iT2) '______'])
    
%     if(length(char(num2str(iT1)))>2)
%         subjectsLists(sliceCount,:) = char(['Different_T2_' num2str(iT1) '_________']);
%     else
%         subjectsLists(sliceCount,:) = char(['Different_T2_' num2str(iT1) '_________']);
%     end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end


%For Blood T2
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    subjectsLists(sliceCount,:) = char(['Different_Bp_T2_' num2str(iT2) '______'])
    
%     if(length(char(num2str(iT1)))>2)
%         subjectsLists(sliceCount,:) = char(['Different_T2_' num2str(iT1) '_________']);
%     else
%         subjectsLists(sliceCount,:) = char(['Different_T2_' num2str(iT1) '_________']);
%     end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end  
end

%% Differnt B1

b1Arrs = 0.5:0.1:1.3;
myoLen = length(b1Arrs);
bpLen = length(b1Arrs);
%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 45*ones(1,myoLen);
bpT2 = 200*ones(1,myoLen);

%for T1
%Myo T1 = 1500;
%blood T1 = 2000;
myoT1 = 1500*ones(1,myoLen);
bpT1 = 2000*ones(1,myoLen);

%for off-resonance
offResArrs = 0*ones(1,myoLen);
%for B-9
%b1Arrs =  ones(1,myoLen);
%for inversion pulse
inveArrs = ones(1,myoLen);
%for HR
hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);

% t1Arrs = 1200;

%For Myo
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(60+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    if(length(char(num2str(iB1Scale*10)))>1)
        subjectsLists(sliceCount,:) = char(['Different_Myo_B1_' num2str(iB1Scale*10) '______'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Myo_B1_' num2str(iB1Scale*10) '_______'])
    end
    

    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

%For Blood
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    if(length(char(num2str(iB1Scale*10)))>1)
        subjectsLists(sliceCount,:) = char(['Different_Bp_B1_' num2str(iB1Scale*10) '_______']);
    else
        subjectsLists(sliceCount,:) = char(['Different_Bp_B1_' num2str(iB1Scale*10) '________']);
    end
    

    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end  
end


%% Differnt Off
offResArrs = -125:25:125;
myoLen = length(offResArrs);
bpLen = length(offResArrs);
%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 45*ones(1,myoLen);
bpT2 = 200*ones(1,myoLen);

%for T1
%Myo T1 = 1500;
%blood T1 = 2000;
myoT1 = 1500*ones(1,myoLen);
bpT1 = 2000*ones(1,myoLen);

%for off-resonance
%offResArrs = 0*ones(1,myoLen);
%for B1
b1Arrs = ones(1,myoLen);
%for inversion pulse
inveArrs = ones(1,myoLen);
%for HR
hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);

% t1Arrs = 1200;

%For Myo
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
     
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    
    if(length(char(num2str(ioffFre)))>3)
        subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(ioffFre) '___'])
    elseif(length(char(num2str(ioffFre)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(ioffFre) '____'])
    elseif(length(char(num2str(ioffFre)))>1)
        subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(ioffFre) '_____'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(ioffFre) '______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

%For Bp
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    
    if(length(char(num2str(ioffFre)))>3)
        subjectsLists(sliceCount,:) = char(['Different_Bp_Off_' num2str(ioffFre) '____'])
    elseif(length(char(num2str(ioffFre)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Bp_Off_' num2str(ioffFre) '_____'])
    elseif(length(char(num2str(ioffFre)))>1)
        subjectsLists(sliceCount,:) = char(['Different_Bp_Off_' num2str(ioffFre) '______'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Bp_Off_' num2str(ioffFre) '_______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end


%% Differnt HR
hrArrs = 40:10:120;
myoLen = length(hrArrs);
bpLen = length(hrArrs);
%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 45*ones(1,myoLen);
bpT2 = 200*ones(1,myoLen);

%for T1
%Myo T1 = 1500;
%blood T1 = 2000;
myoT1 = 1500*ones(1,myoLen);
bpT1 = 2000*ones(1,myoLen);

%for off-resonance
offResArrs = 0*ones(1,myoLen);
%for B1
b1Arrs = ones(1,myoLen);
%for inversion pulse
inveArrs = ones(1,myoLen);
%for HR
%hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);

% t1Arrs = 1200;

%For Myo
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    

    if(length(char(num2str(iHR)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Myo_HR_' num2str(iHR) '_____'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Myo_HR_' num2str(iHR) '______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

%For Bp
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    
    if(length(char(num2str(iHR)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Bp_HR_' num2str(iHR) '______'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Bp_HR_' num2str(iHR) '_______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end


%%

%%Differnt inversion pulse
inveArrs = 0.9:0.01:1;
myoLen = length(inveArrs);
bpLen = length(inveArrs);
%for T2
%45 for Myocardaium
%200 for blood
myoT2 = 45*ones(1,myoLen);
bpT2 = 200*ones(1,myoLen);

%for T1
%Myo T1 = 1500;
%blood T1 = 2000;
myoT1 = 1500*ones(1,myoLen);
bpT1 = 2000*ones(1,myoLen);

%for off-resonance
offResArrs = 0*ones(1,myoLen);
%for B1
b1Arrs = ones(1,myoLen);
%for inversion pulse
%inveArrs = ones(1,myoLen);
%for HR
hrArrs = 60*ones(1,myoLen);
%for SNR
minimalSNR = 100;
maximumSNR = 100;
snrArrs = 100*ones(1,myoLen);

% t1Arrs = 1200;

%For Myo
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = myoT1(ix);
    iT2 = myoT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    %invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    

    if(length(char(num2str(invEff*100)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Myo_IR_' num2str(invEff*100) '_____'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Myo_IR_' num2str(invEff*100) '______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

%For Bp
for ii = 1: myoLen
    
    ix = ii;
    
    %1. Different HR, T1 and T2
    iT1 = bpT1(ix);
    iT2 = bpT2(ix);
    ioffFre = offResArrs(ix);
    iB1Scale = b1Arrs(ix);
    invEff = inveArrs(ix);
    iHR = hrArrs(ix);
    iSNR = snrArrs(ix);
    
    HRs = round(iHR+ normrnd(0,2,1,15));
    expT2 = exp(-InvDur/iT2);
    %simInv = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1, iT2)
    %invEff = abs(expT2);
    
    clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(invEff), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
    
    if(oflg)
        
        for iSx = 1:Sx
            
            for iSy = 1: Sy
                %
                %Rice distribution noise
                clear noiseSig;
                normalizationSig4SNR = max([max(abs(oSig(:,1)+1i*oSig(:,2)))]);
                istd =normalizationSig4SNR/iSNR;
                %add noise
                noiseSig = randn(16,1);
                noiseSig = noiseSig/std(noiseSig);
                noiseSig = noiseSig-mean(noiseSig);
                noiseSig = noiseSig*istd;
                
                phInfo = oSig(:,3); % angle(oSig(:,1)+ 1i*oSig(:,2));
                sigPolarity = phInfo./abs(phInfo).*ones(8,1)*phInfo(end)./abs(phInfo(end));
                
                inputSigWithoutNoise = abs(oSig(:,1) + 1i*(oSig(:,2))).*sigPolarity;
                %Normalization
                inputSigWithoutNoise = inputSigWithoutNoise./max(abs(inputSigWithoutNoise)*1.1);
                
                inputSigWithNoise = abs(oSig(:,1)+noiseSig(1:8) + 1i*(oSig(:,2)+noiseSig(9:16))).*sigPolarity;
                %Normalization
                inputSigWithNoise = inputSigWithNoise./(max(abs(inputSigWithNoise))*1.1);
                
                
                T1w_noisy(:, iSx, iSy) = inputSigWithNoise';
                Ti(:, iSx, iSy) = [oTinv' oRecoveryPeriod].*TimescalingFactor;
                T1(:, iSx, iSy) = iT1;
                %RecovreyPeriod(ii) = oRecoveryPeriod;
                [FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv(1:8));
                FitT1_noisy(:, iSx, iSy) = FitT1_star_noisy*(FitB/FitA-1).*TimescalingFactor;
                
                
                [FitA,FitB,FitT14_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:4,1), oTinv(1:4));
                FitT14_noisy = FitT14_noisy_star*(FitB/FitA-1);
                T14_noisy(:, iSx, iSy) = FitT14_noisy.*TimescalingFactor;
            end
        end
        
    end
      
    PreMOLLIT1wTIs(sliceCount,1,1:T1wNum,:,:) = T1w_noisy;
    PreMOLLIT1wTIs(sliceCount,1,T1wNum+1:2*T1wNum+1,:,:) = Ti;
    PreMOLLIT1MapOffLine(sliceCount,:,:) = FitT1_noisy;   
    PreMOLLIT1MapOnLine(sliceCount,:,:) = FitT1_noisy;    
    SASHA2PT1MapOffLine(sliceCount,:,:) = T1;
    SASHA3PT1MapOffLine(sliceCount,:,:) = T1;
    PreMOLLIMyoMask(sliceCount,1:Sx,1:Sy) = 1;
    PreMOLLIBPMask(sliceCount,1:Sx,1:Sy) = 1;
    Pre5HBsT1Fit(sliceCount,1:Sx,1:Sy) = T14_noisy;
    
    %subjectsLists(sliceCount,:) = char(['Different_Myo_Off_' num2str(iB1Scale*10) '___']);
    
    if(length(char(num2str(invEff*100)))>2)
        subjectsLists(sliceCount,:) = char(['Different_Bp_IR_' num2str(invEff*100) '______'])
    else
        subjectsLists(sliceCount,:) = char(['Different_Bp_IR_' num2str(invEff*100) '_______'])
    end
    
    
    sliceCount = sliceCount+1;
    %
    if(mod((ii/totalSamples*100),10)==0)
        str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
        disp(str);
    end
    
end

% build data here

Pre5HBsT1wTIs = PreMOLLIT1wTIs;
Pre5HBsBPMask = PreMOLLIMyoMask;
Pre5HBsMyoMask = PreMOLLIMyoMask;
Pre5HBsSepMask = PreMOLLIMyoMask;
PreMOLLISepMask = PreMOLLIMyoMask;
PreSASHABPMask = PreMOLLIMyoMask;
PreSASHAMyoMask = PreMOLLIMyoMask;
PreSASHASepMask = PreMOLLIMyoMask;

PreSASHA2PT1MapOffLine = SASHA2PT1MapOffLine.*TimescalingFactor;
PreSASHA3PT1MapOffLine = SASHA3PT1MapOffLine.*TimescalingFactor;

save([matPath filesep 'allSim_pre_T1map_50_50_SL5_Testing.mat'], ...
    'Pre5HBsT1wTIs',...
    'Pre5HBsBPMask',...
    'Pre5HBsMyoMask',...
    'Pre5HBsSepMask',...
    'PreMOLLIT1wTIs', ...
    'PreMOLLIT1MapOffLine', ...
    'PreMOLLIT1MapOnLine',...
    'PreMOLLIMyoMask', ...
    'PreMOLLIBPMask', ...
    'PreMOLLISepMask',...
    'Pre5HBsT1Fit', ... 
    'PreSASHA2PT1MapOffLine',...
    'PreSASHA3PT1MapOffLine',...
    'PreSASHABPMask',...
    'PreSASHAMyoMask',...
    'PreSASHASepMask',...
    'subjectsLists');
