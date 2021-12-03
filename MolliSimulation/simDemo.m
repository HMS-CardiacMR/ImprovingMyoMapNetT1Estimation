
clear all
close all;
HR = 60; %bpm
T1s=1000:100:1500; %
T2=40;   %

%1. Different HR, T1 and T2
for ix =1:length(T1s)
    T1 = T1s(ix);
    [oflag, oSig, oTinv]= MOLLI53Sim(HR, T1, T2);
    if(oflag)
        [FitA,FitB,FitT1] = MOLLIT1Fitting(oSig(:,3), oTinv);
    end
    fitT1s(ix) = FitT1;
end


figure, hold on; xlim([00 1500]); ylim([00 1500]); plot(T1s,fitT1s,'*'); hold off;

%2. Different HR, T1 and T2
% you can specify the heart rate of each cardiac cycle, e.g., HR from 40 to 120 bpm
HRs = [40:8:120]; % the length of HRs must be equal to the 11 (5+(3)+3)

[oflag, oSig, oTinv]= MOLLI53Sim(HRs, T1, T2);
if(oflag)
    [FitA,FitB,FitT1] = MOLLIT1Fitting(oSig(:,3), oTinv);
end

%3. Different inversion efficiency, e.g., from 0.7 to 1
HR =60;
invEff = 0.7;
[oflag, oSig, oTinv]= MOLLI53Sim(HRs, T1, T2, invEff);
if(oflag)
    [FitA,FitB,FitT1] = MOLLIT1Fitting(oSig(:,3), oTinv);
end


%4. Different off resonance, e.g., from -100 to 100 Hz
HR =60;
invEff = 1;
ioffFre = 50;
[oflag, oSig, oTinv]= MOLLI53Sim(HRs, T1, T2, invEff,ioffFre);
if(oflag)
    [FitA,FitB,FitT1] = MOLLIT1Fitting(oSig(:,3), oTinv);
end

%5.Different B1 scale, e.g., from 0.6 to 1.2
HR =60;
invEff = 1;
ioffFre = 0;
iB1Scale = 0.77;

[oflag, oSig, oTinv]= MOLLI53Sim(HRs, T1, T2, invEff,ioffFre,iB1Scale);
if(oflag)
    [FitA,FitB,FitT1] = MOLLIT1Fitting(oSig(:,3), oTinv);
end

%6. Difffernr SNR.
HR = 60; %bpm
T1=1500; %
T2=42;   %
[oflag, oSig, oTinv]= MOLLI53Sim(HR, T1, T2);
if(oflag)
    %for SNR = 20;
    iSNR = 40;
    normalizationSig4SNR = max(abs(oSig(:,3)));
    istd =normalizationSig4SNR/iSNR;
    
    %add noise
    noiseSig = randn(8,1);
    noiseSig = noiseSig/std(noiseSig);
    noiseSig = noiseSig-mean(noiseSig);
    noiseSig = noiseSig*istd;
    inputSIg = oSig(:,3)+noiseSig;
    
    [FitA,FitB,FitT1] = MOLLIT1Fitting(inputSIg(:), oTinv(:));
end
