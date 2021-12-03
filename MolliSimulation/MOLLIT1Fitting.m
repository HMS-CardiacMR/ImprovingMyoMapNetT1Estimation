  
function [oaEst,obEst,oT1]=MOLLIT1Fitting(Sig, iTinv)
%Fit T1 and T2
data = Sig(:);
extra.tdVec = iTinv(:);

%x0 = [max(Sig), -2*max(Sig),1200];
x0 = [max(abs(Sig)), -2*max(abs(Sig)),1000];
%% Do the fit
x = fminsearch( ...
  @(x)sum(abs( data-( x(1)-x(2)*exp(-extra.tdVec/x(3))) ).^2), ...
  x0,optimset('display','off'));

aEst = x(1);
bEst = x(2);
T1 = x(3); %*(bEst/aEst-1);

oaEst = aEst;
obEst = bEst;
oT1 = T1;
end