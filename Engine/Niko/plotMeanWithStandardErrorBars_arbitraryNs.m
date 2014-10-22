function [mean_y,se_mean_y]=plotMeanWithStandardErrorBars_arbitraryNs(x,y,fig,titleString,col,min_n_OR_n_bins)

% USAGE
%       [mean_Y,se_Y]=plotMeanWithStandardErrorBars_arbitraryNs(x,y[,fig,title,color,min_n])
%
% FUNCTION
%       plots the mean of y as a function of x with error
%       bars indicating the standard error of the mean.
%
% ARGUMENTS
% x     vector of n x-axis coordinates
%
% y     vector of y-axis coordinates whose number and order corresponds to x
%
% fig   optional argument. if fig is...
%       - unspecified: a fresh figure with index 845 is drawn
%       - a scalar: a fresh figure with index fig is drawn
%       - a quadruple: a fresh subplot(fig(2),fig(3),fig(4)) is added to
%           figure fig(1)
%       - a quintuple: as for quadruple, but the curved are drawn into the
%           existing axes if fig(5)==1
%
% titleString
%       optional figure title
%
% col   optional color specification (defaults to black) as RGB triple
%
% min_n optional minimum required number of values of y per unique value of x. if
%       the actual number of values of y is smaller than min_n, the next
%       unique x value is included until at least min_n y values have been
%       accumulated for averaging. min_n defaults to 12.
%       (see function summarize_min_n.m)
%
% RETURNS
% mean_Y
%       the means across the vertical dimension (observations) of Y
%
% se_mean_Y
%       the standard errors of the means mean_Y when they are considered
%       estimates of the population means mu.
%
% nk, 1 jun 2006

if ~exist('fig'), fig=1; end
if ~exist('titleString'), titleString=''; end
if ~exist('col'), col=[0 0 0]; end

if length(fig)>3
    h=figure(fig(1)); subplot(fig(2),fig(3),fig(4));
else
    h=figure(fig(1)); clf;
end
set(h,'Color','w');

if ~exist('min_n_OR_n_bins','var')
    min_n_OR_n_bins=12;
end

%[xp,mean_yp,se_mean_yp]=summarize_min_n(x,y,min_n)
[xp,mean_yp,se_mean_yp]=summarize_n_bins(x,y,min_n_OR_n_bins)


% x=x(:);
% y=y(:);
% 
% if length(y)~=length(x)
%     error('plotMeanWithStandardErrorBars_arbitraryNs: x and y must have the same number of elements.');
% end
% 
% if size(x,1)>size(x,2)
%     x=x'; % make a row
% end
% 
% if size(y,1)>size(y,2)
%     y=y'; % make a row
% end    
%     
% sortedSet_x=unique(x);
% 
% mean_y=nan(size(sortedSet_x));
% std_y=nan(size(sortedSet_x));
% n_y=nan(size(sortedSet_x));
% 
% i=1;
% for cx=sortedSet_x
%     cys=y(find(x==cx));
%     mean_y(i)=mean(cys,1);
%     std_y(i)=std(cys,0,1); % unbiased estimator of population std (divide by n-1) along vertical dim (last argument)
%     n_y(i)=length(cys);
%     i=i+1;
% end
% 
% 
% se_mean_y=std_y./sqrt(n_y); 
% % var_Y = std_Y.^2
% % var_sum_Y = var_Y*n
% % std_sum_Y = sqrt(var_Y*n)
% % std_mean_Y = std_sum_Y/n = sqrt(var_Y*n)/n = sqrt(var_Y)*sqrt(n)/n = sqrt(var_Y)/sqrt(n) = std_Y/sqrt(n);
    

if length(fig)<5 | (length(fig)==5 & fig(5)==0)
    hold off;
else
    hold on;
end

lw=3;
errorbar(xp,mean_yp,se_mean_yp,'Color',col,'LineWidth',lw);
title(titleString);



