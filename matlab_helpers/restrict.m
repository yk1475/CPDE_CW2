function coarse = restrict(fine) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  restriction routine (full weighting) 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [r,c] = size(fine);
   n = r-1;
   m = c-1; 
   n2 = n/2;
   m2 = m/2;

   coarse = zeros(n2+1,m2+1); 
   coarse(2:n2,2:m2) =  1/4*fine(3:2:n-1, 3:2:m-1) + ...
                         1/8*fine(2:2:n-2, 3:2:m-1) + ...
                         1/8*fine(4:2:n,   3:2:m-1) + ...
                         1/8*fine(3:2:n-1, 2:2:m-2) + ...
                         1/8*fine(3:2:n-1, 4:2:m)   + ...
                         1/16*fine(2:2:n-2,2:2:m-2) + ...
                         1/16*fine(2:2:n-2,4:2:m)   + ...
                         1/16*fine(4:2:n,  2:2:m-2) + ...
                         1/16*fine(4:2:n,  4:2:m);

%   this line alone would be the simple mapping
  % coarse(2:n2,2:m2) =  fine(3:2:n-1, 3:2:m-1);


