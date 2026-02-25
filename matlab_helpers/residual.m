   function res = residual(u,f) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
% Residuals: What is f-Au over the grid (1/m, 1/n)? %
%                                                   %   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [rows,columns] = size(u);
   n = rows-1;
   m = columns-1; 
   n2 = n*n; %dy = 1/n;
   m2 = m*m; %dx = 1/m;
   m2n22 = 2*(m2 + n2); 
   res = zeros(size(u)); 
   for i=2:n
   for j=2:m
   res(i,j) = f(i,j)+u(i,j)*m2n22 -(u(i,j+1)+u(i,j-1))*m2 -(u(i+1,j)+u(i-1,j))*n2; 
   end
   end