function fine = interpolate(coarse)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%  interpolation routine (inverse full weighting)   % 
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [r,c] = size(coarse);
   n = r-1;
   m = c-1; 
   n2 = 2*n;
   m2 = 2*m; 

   fine = zeros(n2+1,m2+1);

   fine(3:2:n2-1,3:2:m2-1) =  coarse(2:n,2:m); 
   fine(3:2:n2-1,2:2:m2)   = (coarse(2:n,1:m) + coarse(2:n,2:m+1))/2;
   fine(2:2:n2  ,3:2:m2-1) = (coarse(1:n,2:m) + coarse(2:n+1,2:m))/2;
   fine(2:2:n2  ,2:2:m2)   = (coarse(1:n,1:m) + coarse(2:n+1,2:m+1) + ...
                              coarse(2:n+1,1:m) + coarse(1:n,2:m+1) )/4;
