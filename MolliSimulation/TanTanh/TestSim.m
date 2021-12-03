%sim Test
close all;
clear all;
SIEMENS_TanTanh = load('SIEMENSTanTtanh.csv');

%[oInvEff]=TanTanhSim(SIEMENS_TanTanh,1, 0, 1500, 40)
B1 = 0:0.01:2;


off = -500:10:500;

for ix = 1:length(B1)
    for ij = 1:length(off)
        invMap(ix,ij) = TanTanhSim(SIEMENS_TanTanh,B1(ix), off(ij), 1500, 40);
    end
end

figure,contour(1:201,1:101,invMap',[-1:0.05:-0.7],'ShowText','on');

iT2 = 40:5:250;
for ij = 1:length(iT2)
    invMapT2(ij) = TanTanhSim(SIEMENS_TanTanh,1, 0, 1500, iT2(ij));
end


figure,hold on; axis([40 250 0.9 1]); grid on;
plot(iT2,exp(-2.56./iT2),'g--','linewidth',3);
plot(iT2,abs(invMapT2),'r','linewidth',3);title('different T2');
hold off;



iT1 = 100:100:2000;
for ij = 1:length(iT1)
    invMapT1(ij) = TanTanhSim(SIEMENS_TanTanh,1, 0, iT1(ij), 200);
end


figure,hold on; axis([100 2000 0.9 1]); grid on;
plot(iT1,abs(invMapT1),'linewidth',3);title('different T1');
hold off;