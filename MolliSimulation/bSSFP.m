%[oMxyz] = bSSFP(iFAs, iTR, iTFEFactor, iT1, iT2, iMzInit, ioffFre)
%bSSFP readout for generating magnetization (Mx,My,Mz) of TFEFactor K-space lines, including ramp-up pulses
%
%[oMxyz] = bSSFP(iFAs, iTR, iTFEFactor, iT1, iT2, iMzInit)
%[oMxyz] = bSSFP(iFAs, iTR, iTFEFactor, iT1, iT2, iMzInit, ioffFre)
%
%Input:
%   iFA=    specified flip angle of each K-space lines
%   iTR=    repated time in ms
%   TFEFactor=  number of K-space lines
%   iMzInit= inital Mz
%   ioffFre: frequence (Hz) of off resonance
%Output:
%   oMxyz=  [Mx, My, Mz]

function [oMxyz] = bSSFP(iFAs, iTR, iTFEFactor, iT1, iT2, iMzInit, ioffFre)

if nargin < 6 || isempty(iMzInit)
    iMzInit = 1;     
end 

if nargin < 7 || isempty(ioffFre)
    ioffFre = 0;     %frequeence (Hz) of off-resonance, 
end 

FA = iFAs;
TR = iTR;
TFEFactor = iTFEFactor;
T1 = iT1;
T2 = iT2;
MzInit = iMzInit;
offFre = ioffFre;

% phase 
PHI = pi+ 2*pi*offFre*TR/1000;  %  PHI shoule be set as pi with angle accnounting for off resonance

%T1 and T2 relaxation
E_1 = exp(-TR/T1);
E_2 = exp( -TR/T2);

%% basic matrix
M_0 = [ 0;  ...
    0;  ...
    MzInit] ;

M_eq = [ 0;  ...
    0;  ...
    1] ;
%rotating matrix
P = [ E_2*cos(PHI) E_2*sin(PHI) 0;  ...
    -E_2*sin(PHI) E_2*cos(PHI) 0; ...
    0           0           E_1];


%%initial value
M_r_0negative = M_0 ;
MxyzArr = zeros(3,TFEFactor);

for i = 1:TFEFactor
    
    ALPHA = FA(i);
    R_x_alpha = [ 1     0         0;...
        0 cos(ALPHA) sin(ALPHA);  ...
        0 -sin(ALPHA) cos(ALPHA)];
    
    M_r_0positive = R_x_alpha*M_r_0negative ;
    M_r_0negative = P*M_r_0positive+ ( 1- E_1 )*M_eq ;
    
    
    %Here, we assumed that Te=TR/2
    MxyzArr(1,i ) = M_r_0positive(1,1)*E_2*cos(PHI) + M_r_0positive(2,1)*E_2*sin(PHI) ;
    MxyzArr(2,i ) = -M_r_0positive(1,1)*E_2*sin(PHI) + M_r_0positive(2,1)*E_2*cos(PHI) ;
    MxyzArr(3,i ) =  M_r_0positive(3,1)*E_1+( 1- E_1)*M_eq(3,1);
end
oMxyz = MxyzArr;
end