function [Mx,My,Mz]=rot_matrix_Freq(B1xy_m,T,Freq_v,M, T1, T2)

RF_NPoints=size(B1xy_m,2);
deltaT=T/RF_NPoints*1e-3; % s
RFr=2*pi*B1xy_m(1,1:RF_NPoints);  % 2*pi*Hz
phs=B1xy_m(2,1:RF_NPoints); % in degree
offset=2*pi*Freq_v';   % 2*pi*Hz % Freq_v is only the B0 inhomogneity
                       % the FM function is replaced by phase modulated
                       % function incoporated in the phase 



max=size(Freq_v,2);
i2 = 1:max;

E_T1 = exp(-deltaT*1000/T1);
E_T2 = exp(-deltaT*1000/T2);
Meq = M( (i2-1)*3+3 );

Mtemp = zeros(3*max,3);     % a matrix (3*max,3)
Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );          
Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 );          
Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);

for i1 = 1:RF_NPoints;
  % not good for multiple offsets: 
   %alpha=sqrt(RFr(i1)^2+offset(i2).^2)*deltaT; % flip angle rotating around x axis
   %phi=atan2(offset(i2),RFr(i1)); % flip angle rotating around y axis
   %theta=(pi/180)*phs(i1); % flip angle rotating around z axis
   %M=zrot(theta)*yrot(phi)*xrot(alpha)*yrot(-phi)*zrot(-theta)*M; 
   % reference: Palmer's book P15

  % for offset vector:  
   %zrot(-theta)*M
   theta=(pi/180)*phs(i1); % flip angle rotating around z axis  not spatial dependent---(1,1)
   rot_matrix=zeros(3*max,3);    % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+3, 3 )=1;
   rot_matrix( (i2-1)*3+1, 1 )=cos(-theta);   
   rot_matrix( (i2-1)*3+1, 2 )=-sin(-theta);
   rot_matrix( (i2-1)*3+2, 1 )=sin(-theta);
   rot_matrix( (i2-1)*3+2, 2 )=cos(-theta);
   M=sum(rot_matrix.*Mtemp,2);                % a column vector (3*max,1)
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );    % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 );          
   Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);

   %yrot(-phi)*zrot(-theta)*M
   %offset=gyro*1e3*(xy(i2,1)*GEx_v(i1)+xy(i2,2)*GEy_v(i1)+B0_Gauss);   % 2*pi*Hz
   phi=atan2(offset(i2),RFr(i1)); % flip angle rotating around y axis:  spatial dependent---a column vector (max,1)
   
   rot_matrix=zeros(3*max,3);    % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+2, 2 )=1;
   rot_matrix( (i2-1)*3+1, 1 )=cos(-phi(i2));  % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+1, 3 )=sin(-phi(i2));
   rot_matrix( (i2-1)*3+3, 1 )=-sin(-phi(i2));
   rot_matrix( (i2-1)*3+3, 3 )=cos(-phi(i2));
   M=sum(rot_matrix.*Mtemp,2);                % a column vector (3*max,1)
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );   % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 );          
   Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);

   %xrot(alpha)*yrot(-phi)*zrot(-theta)*M;
   alpha=sqrt(RFr(i1)^2+offset(i2).^2)*deltaT; % flip angle rotating around x axis:  spatial dependent---a column vector (max,1)   
   rot_matrix=zeros(3*max,3);    % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+1, 1 )=1;
   rot_matrix( (i2-1)*3+2, 2 )=cos(alpha(i2));  % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+2, 3 )=-sin(alpha(i2));
   rot_matrix( (i2-1)*3+3, 2 )=sin(alpha(i2));
   rot_matrix( (i2-1)*3+3, 3 )=cos(alpha(i2));
   M=sum(rot_matrix.*Mtemp,2);                   % a column vector (3*max,1)
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );   % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 );          
   Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);

   %yrot(phi)*xrot(alpha)*yrot(-phi)*zrot(-theta)*M;
   rot_matrix=zeros(3*max,3);    % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+2, 2 )=1;
   rot_matrix( (i2-1)*3+1, 1 )=cos(phi(i2));  % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+1, 3 )=sin(phi(i2));
   rot_matrix( (i2-1)*3+3, 1 )=-sin(phi(i2));
   rot_matrix( (i2-1)*3+3, 3 )=cos(phi(i2));
   M=sum(rot_matrix.*Mtemp,2);                   % a column vector (3*max,1)
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );   % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 );          
   Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);

   %zrot(theta)*yrot(phi)*xrot(alpha)*yrot(-phi)*zrot(-theta)*M;
   rot_matrix=zeros(3*max,3);    % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+3, 3 )=1;
   rot_matrix( (i2-1)*3+1, 1 )=cos(theta);   % a matrix (3*max,3)
   rot_matrix( (i2-1)*3+1, 2 )=-sin(theta);
   rot_matrix( (i2-1)*3+2, 1 )=sin(theta);
   rot_matrix( (i2-1)*3+2, 2 )=cos(theta);
   M=sum(rot_matrix.*Mtemp,2);                % a column vector (3*max,1)
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 );    % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 );         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 ); 
   
   
   Mtemp( (i2-1)*3+1, 1 ) = M( (i2-1)*3+1 ).*E_T2;    % a matrix (3*max,3)
   Mtemp( (i2-1)*3+1, 2 ) = M( (i2-1)*3+2 ).*E_T2;         
   Mtemp( (i2-1)*3+1, 3 ) = M( (i2-1)*3+3 ).*E_T1+Meq.*(1-E_T1); 
   
   Mtemp( (i2-1)*3+2, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   Mtemp( (i2-1)*3+3, 1:3 ) = Mtemp( (i2-1)*3+1,1:3);
   
   
   
end
   
Mx(1:max,1)=M(((1:max)-1)*3+1,size(M,2));    % Mx[max, n]
My(1:max,1)=M(((1:max)-1)*3+2,size(M,2));    % My[max, n]
Mz(1:max,1)=M(((1:max)-1)*3+3,size(M,2));    % Mz[max, n]
   
end