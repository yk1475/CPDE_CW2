
function u = GS(uinit,b,Niter)

%------------------------------------------%
% Takes Niter Gauss-Seidel iterations for  %
% the discrete Poisson equation in form    %
% Au=b, on unit square starting from uinit.%
%------------------------------------------%
  
  m = size(b,1);
  n = size(b,2);
  dx = 1/(m-1);
  dy = 1/(n-1);
  m2 = 1/(dx*dx);
  n2 = 1/(dy*dy);
  m2n22 = 2*(m2 + n2); 

  % initialization
  unew = uinit;
  
  % iteration
  for k=1:Niter
    for i=2:m-1
      for j=2:n-1
          % For Gauss-Seidel, overwrite u and use calculated new values
	unew(i,j) = (m2*(unew(i+1,j)+unew(i-1,j)) + ...
	          n2*(unew(i,j+1)+unew(i,j-1)) - ...
	          b(i,j))/m2n22;
      end
    end
  end

  u = unew;
  