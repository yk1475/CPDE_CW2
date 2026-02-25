% Program OMG! to test Multigrid
% 1x1 square, delsqr(u)=1 (or f) with zero Dirichlet conditions
clear;
M=1025; % 2^p+1
N=M
f=ones(M,N); 
for i=1:M
    x=(i-1)/M;
  for j=1:N
      y=(j-1)/N;
    f(i,j)=sin(9*pi*x)*cos(pi*y)+.2*x*y;
 end;
end;
figure(2)
surf(f)
shading interp
colorbar
uinit=zeros(M,N);
tic
%u=SOR(uinit,f,1.8,2500);
 u=MultigridV(uinit,f);
 for k=1:3
     u=MultigridV(u,f);
 %maxerror=max(max(residual(u,f)))
 end
%u=FullMG(f);
%u=FMGV(uinit,f,3);
time=toc
maxerror=max(max(residual(u,f)))
figure(1);
% mesh(u)
pcolor(u);
shading interp;
colorbar
