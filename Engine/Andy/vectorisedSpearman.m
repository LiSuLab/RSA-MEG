function [ rho ] = vectorisedSpearman( A , B )
%spearmans_rho : Find's spearmans rho for two matrices
%   Follows the wikipeadia version

[m n] = size(A);

[AS ASorder] =  sort(A);
ASorder = ASorder + repmat(0:m:(n-1)*m+1,  m , 1);

BS = zeros(m, n);
BS(:) = B(ASorder(:));
% [junk BSorderkey] =  sort(BS);
% BSorder = (repmat((1:m)',1,n));
% BSorder(:) = BSorder(BSorderkey(:));
[junk BSorder] = sort(BS); %
[junk BSrank] = sort(BSorder); %

% dsqr = (repmat((1:m)', 1, n)-BSorder).^2;
dsqr = (repmat((1:m)', 1, n)-BSrank).^2; %
sizeofp = sum(~isnan(A));
rho = 1-((6*nansum(dsqr))./ (sizeofp.*(sizeofp.^2-1)));

end