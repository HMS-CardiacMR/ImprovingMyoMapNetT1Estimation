
%% for pre-contrast
clear all 

savePath = '/mnt/alp/Users/RuiGuo/BsNetMOLLI/Training/Numerical_simulation';
dugDisp = 0;

%for Simulating the Tan/Tanh pulse
SIEMENS_TanTanh = load('./TanTanh/SIEMENSTanTtanh.csv');
Os = 'Os_Spider';

%% for myocardium
totalSamples = 1000000;
t1RandomIx = randperm(totalSamples);
t2RandomIx = randperm(totalSamples);
offResoanceRandomIx = randperm(totalSamples);
b1RandomIx = randperm(totalSamples);
inversioneffiencyRandomIx = randperm(totalSamples);
hrRandomIx = randperm(totalSamples);
snrRandomIx = randperm(totalSamples);

%1. For T1
t1MU = 2500;
t1SD = 100;
%t1Arrs = normrnd(t1MU,t1SD,totalSamples,1);
minimalT1 = 1000;
maximumT1 = 2000;

minimalT1 = 100;
maximumT1 = 2500;
t1Arrs =  minimalT1+(maximumT1-minimalT1)*(rand(totalSamples,1));
[counts,centers] = hist(t1Arrs,300000);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('T1 Distribution'); 
    xlabel('T1'); 
    ylabel('Number');
    hold off;
end

%2. For T2
t2MU = 45;
t2SD = 7.5;
t2Arrs = normrnd(t2MU,t2SD,totalSamples,1);
t2Arrs(t2Arrs<10) = t2MU;

minimalT2 = abs(min(t2Arrs));
maximumT2 = abs(max(t2Arrs));

minimalT2 = 20;
maximumT2 = 80;

minimalT2 = 20;
maximumT2 = 250;
t2Arrs =  minimalT2+(maximumT2-minimalT2)*(rand(totalSamples,1));
minimalT2 = abs(min(t2Arrs));
maximumT2 = abs(max(t2Arrs));
[counts,centers] = hist(t2Arrs,120);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('T2 Distribution'); 
    xlabel('T2 [ms]'); 
    ylabel('Number');
    hold off;
end

%3. For off-resonance
offResMU = 0;
offResSD = 20;  %original is 25(used in paper)    large 40 , minimal 20
offResSD = 30;
offResArrs = normrnd(offResMU,offResSD,totalSamples,1);

% minimalOFF = -130;
% maximumOFF = 130;
% offResArrs =  minimalOFF+(maximumOFF-minimalOFF)*(rand(totalSamples,1));
minimalOFF = abs(min(offResArrs));
maximumOFF = abs(max(offResArrs));
[counts,centers] = hist(offResArrs,250);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('Off-resonance Distribution'); 
    xlabel('Off-resonance [Hz]'); 
    ylabel('Number');
    hold off;
end

%4. For B1
b1MU = 1;
b1SD = 0.075;  %original is 0.1(used in paper)  large 0.15 small 0.075 (4 Tissues simulation)
b1SD = 0.2; 
b1Arrs = normrnd(b1MU,b1SD,totalSamples,1);
b1Arrs(b1Arrs<0.3) = 1;
% 
minimalB1 = min(b1Arrs);
maximumB1 = max(b1Arrs);

% minimalB1 = min(0.5);
% maximumB1 = max(1.5);
% b1Arrs =  minimalB1+(maximumB1-minimalB1)*(rand(totalSamples,1));
[counts,centers] = hist(b1Arrs,300);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('B1 Distribution'); 
    xlabel('B1 [a.u.]'); 
    ylabel('Number');
    hold off;
end

%%5. For inversion efficiency
inveMU = 0.02;
inveSD = 0.01;
inveArrs = 1-abs(normrnd(inveMU,inveSD,totalSamples,1));


minimalInv = min(inveArrs);
maximumInv = max(inveArrs);
%inveArrs =  minimalInv+(maximumInv-minimalInv)*(rand(totalSamples,1));

[counts,centers] = hist(inveArrs,200);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('Inversion-efficiency Distribution'); 
    xlabel('Inversion-efficiency [a.u.]'); 
    ylabel('Number');
    hold off;
end

%%6. For HR
minimalHR = 30;
maximumHR = 130;
hrArrs = minimalHR+(maximumHR-minimalHR)*(rand(totalSamples,1));
[counts,centers] = hist(hrArrs,200);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('HR Distribution'); 
    xlabel('HR [bpm]'); 
    ylabel('Number');
    hold off;
end



%%7. For SNR
minimalSNR = 80;
maximumSNR = 120;
snrArrs = minimalSNR+(maximumSNR-minimalSNR)*(rand(totalSamples,1));
[counts,centers] = hist(snrArrs,200);
if(dugDisp)
    figure, hold on;
    grid on;
    bar(centers,counts);
    title('SNR Distribution'); 
    xlabel('SNR [db]'); 
    ylabel('Number');
    hold off;
end

%% 
Ti = [];
T1w_noisy = [];
T1 = [];

A_noisy = [];
B_noisy = [];
T1_star_noisy = [];
T1_noisy = [];

T15_noisy = [];

data = [];
sliceNum = 1;    
FileName = ['T1_' num2str(minimalT1) '_' num2str(maximumT1) ...
    '_T2_' num2str(minimalT2) '_' num2str(maximumT2) ...
    '_B1_' num2str(minimalB1*10) '_' num2str(maximumB1*10) ...
    '_HR_' num2str(minimalHR) '_' num2str(maximumHR) ...
    '_SNR_' num2str(minimalSNR) '_' num2str(maximumSNR) ...
    '_OFF_' num2str(minimalOFF) '_' num2str(maximumOFF)...
    'SliceProfils_' num2str(sliceNum)];

invDur = 2.56; %ms

SL = 1;
for ii = 1: totalSamples
    
    ix = t1RandomIx( ii );

    %1. Different HR, T1 and T2
     iT1 = t1Arrs(ix);
     iT2 = t2Arrs(ix);
     ioffFre = offResArrs(ix);
     iB1Scale = b1Arrs(ix);
     invEff = inveArrs(ix);
     iHR = hrArrs(ix);
     iSNR = snrArrs(ix);
            
     HRs = round(iHR+ normrnd(0,2,1,15));    
     %simInv = TanTanhSim(SIEMENS_TanTanh,iB1Scale, ioffFre, iT1, iT2); 
      invEff = 1;
     simInv = invEff*exp(-invDur/iT2);
     clear isSLice;
     for is = 1:sliceNum
        sliceProfile = exp(-(is-1).^2/(2*4.^2));
        [oflg, oSig, oTinv,oRecoveryPeriod] = MOLLI53Sim(HRs, iT1, iT2, abs(simInv), ioffFre, iB1Scale*sliceProfile);
        isSLice(:,:,is) = oSig;
     end
     oSig = mean(isSLice,3);
     if(oflg)

%         normalizationSig4SNR = max(abs(oSig(:,3)));
%         istd =normalizationSig4SNR/iSNR;
%         %add noise
%         noiseSig = randn(8,1);
%         noiseSig = noiseSig/std(noiseSig);
%         noiseSig = noiseSig-mean(noiseSig);
%         noiseSig = noiseSig*istd;
%         
%         
        %Rice distribution noise 
        
        normalizationSig4SNR = max([abs(oSig(:,1)+1i*oSig(:,2))]);
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
        
        
        T1w_noisy(SL,:) = inputSigWithNoise';
        Ti(SL, :) = [oTinv' oRecoveryPeriod];
        T1(SL) = iT1;
        %RecovreyPeriod(ii) = oRecoveryPeriod;
        
%         [FitA,FitB,FitT1_star] = MOLLIT1Fitting(inputSigWithoutNoise, oTinv);        
%         FitT1 = FitT1_star*(FitB/FitA-1);
%         T1(ii) = FitT1;
        
            %[FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv);
            FitT1_noisy =0;  %FitT1_star_noisy*(FitB/FitA-1);
            T1_noisy(SL) = FitT1_noisy;
            %T1_star_noisy(ii) = FitT1_star_noisy;
            A_noisy(SL) =0; % FitA;
            B_noisy(SL) =0; % FitB;
         SL = SL+1;
         
        T1w_noisy(SL,:) = inputSigWithoutNoise';
        Ti(SL, :) = [oTinv' oRecoveryPeriod];
        T1(SL) = iT1;
        %RecovreyPeriod(ii) = oRecoveryPeriod;
        
%         [FitA,FitB,FitT1_star] = MOLLIT1Fitting(inputSigWithoutNoise, oTinv);        
%         FitT1 = FitT1_star*(FitB/FitA-1);
%         T1(ii) = FitT1;
        
            %[FitA,FitB,FitT1_star_noisy] = MOLLIT1Fitting(inputSigWithNoise, oTinv);
            FitT1_noisy =0;  %FitT1_star_noisy*(FitB/FitA-1);
            T1_noisy(SL) = FitT1_noisy;
            %T1_star_noisy(ii) = FitT1_star_noisy;
            A_noisy(SL) =0; % FitA;
            B_noisy(SL) =0; % FitB;
         SL = SL+1;
%         [FitA,FitB,FitT15_noisy_star] = MOLLIT1Fitting(inputSigWithNoise(1:5,1), oTinv(1:5));
%         FitT15_noisy = FitT15_noisy_star*(FitB/FitA-1);
%         T15_noisy(ii) = FitT15_noisy; 
       

                
     end
    
     if(mod((ii/totalSamples*100),10)==0)
         str = sprintf('Done--------%d%%',fix(ii/totalSamples*100));
         disp(str);
     end
        
end
%save data
rangeLable = (T1>0).*(T1<3000);

data.T1 = T1(rangeLable>0);
% data.T1_star_noisy =T1_star_noisy(rangeLable>0);
data.A_noisy = A_noisy(rangeLable>0);
data.B_noisy = B_noisy(rangeLable>0);
data.T1_noisy = T1_noisy(rangeLable>0);
% data.T15_noisy = T15_noisy(rangeLable>0);
data.T1w_noisy = T1w_noisy(rangeLable>0,:);
data.Ti = Ti(rangeLable>0,:);
%data.RecovreyPeriod = RecovreyPeriod(rangeLable>0,:);

% save([savePath filesep 'Pre_myo' FileName '_ExpT2_INV1_' num2str(totalSamples)  '.mat'], 'data','-v7.3');

save([savePath filesep 'All' FileName '_ExpT2_INV1_' num2str(totalSamples)  '_2.mat'], 'data','-v7.3');



