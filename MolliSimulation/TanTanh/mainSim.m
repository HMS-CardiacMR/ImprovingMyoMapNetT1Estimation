

% profild TanTanh  profils from Simens IDEA
% PULSENAME:	tan_tanh.tan_tanh_2560us_9500Hz_z10_tk22	
% COMMENT:	tan_tanh_2560us_9500Hz_zeta10_tank22	
% REFGRAD:	1	
% MINSLICE:	10	
% MAXSLICE:	400	
% AMPINT:	272.2109963	
% POWERINT:	485.4999649	
% ABSINT:	493.3085423	
close all;
clear all;
SIEMENS_TanTanh = load('SIEMENSTanTtanh.csv');

B1Profile = SIEMENS_TanTanh(:,1);  %normalized B1
figure, plot(B1Profile,'LineWidth',3); title('B1(t)', 'fontsize',20);
phaseProfile = SIEMENS_TanTanh(:,2);  %rad
figure, plot(phaseProfile,'LineWidth',3); title('\phi(t)','fontsize',20);
figure, plot(diff([phaseProfile]),'LineWidth',3);title('\DeltaW','fontsize',20);

B1x_v=B1Profile.*cos(phaseProfile); % uT
B1y_v=B1Profile.*sin(phaseProfile); % uT
figure, plot(B1x_v,'LineWidth',3); title('B1x','fontsize',20);
figure, plot(B1y_v,'LineWidth',3); title('B1y','fontsize',20);
%%
Tp = 2.56; %ms
B1Max = 14.7; %uT
B1Profile = B1Profile*B1Max;
AM_v = B1Profile;
PM_integ_v = phaseProfile;

B1xy_m(1,:)=AM_v*42.57;  % Hz; 1 uT = 42.57 Hz
B1xy_m(2,:)=PM_integ_v/pi*180; % in degree
delta_omega = 9500; %hz

RF_BWFAC=2*delta_omega*Tp*1e-3; 

BW_Half=[-delta_omega,delta_omega];
Freq_max=500; %RF_BWFAC*1000/T*2; % Hz %input('the maximum absolute frequency [10000 Hz]: ');
Freq_N=1; %input('number of Frequency points [128]: ');
if isempty(Freq_N)
    Freq_N = 1000;
end;
deltaFreq=2*Freq_max/Freq_N;
Freq_v=(-Freq_max+deltaFreq):deltaFreq:Freq_max; % Hz

M0=1;
M=repmat([0 0 M0]',Freq_N,1);

relativeB1 = 0:0.005:2;
MzMatrix = zeros(length(relativeB1), length(Freq_v));

for ix = 1:length(relativeB1)
    [Mx,My,Mz]=rot_matrix_Freq(B1xy_m*relativeB1(ix),Tp,Freq_v,M, 1500, 40);  % rotation matrix method or Bloch equation
    MzMatrix(ix,:) = Mz';
end



