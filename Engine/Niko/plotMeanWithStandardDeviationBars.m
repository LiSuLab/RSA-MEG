function [mean_Y,std_Y]=plotMeanWithStandardDeviationBars(x,Y,fig,titleString,col,barNOTplot)

% USAGE
%       [mean_Y,se_Y]=plotMeanWithStandardErrorBars(x,Y[,fig,title,color,barNOTplot])
%
% FUNCTION
%       plots the mean of row vectors Y against row vector x with error
%       bars indicating the standard error of the mean.
%
% ARGUMENTS
% x     
%       row vector of n x-axis coordinates
%
% Y     
%       matrix whose rows are y-axis coordinates whose number and order
%       corresponds to x
%
% fig   
%       optional argument. if fig is...
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
% col
%       optional color specification (defaults to black) as RGB triple
%
% barNOTplot
%       if this optional argument defaulting to false is set to true, a
%       bargraph instead of a plot is drawn.
%
% RETURNS
% mean_Y
%       the means across the vertical dimension (observations) of Y
%
% se_mean_Y
%       the standard errors of the means mean_Y when they are considered
%       estimates of the population means mu.
%
% nk, 3 sep 2005

if ~exist('fig','var'), fig=1; end
if ~exist('titleString','var'), titleString=''; end
if ~exist('col','var'), col=[0 0 0]; end
if ~exist('barNOTplot','var'), barNOTplot=false; end

if length(fig)>3
    h=figure(fig(1)); subplot(fig(2),fig(3),fig(4));
else
    h=figure(fig(1)); clf;
end
set(h,'Color','w');

n=size(Y,1);

mean_Y=mean(Y,1);

std_Y=std(Y,0,1); % unbiased estimator of population std (divide by n-1) along vertical dim (last argument)

% se_mean_Y=std_Y/sqrt(n); 
% var_Y = std_Y.^2
% var_sum_Y = var_Y*n
% std_sum_Y = sqrt(var_Y*n)
% std_mean_Y = std_sum_Y/n = sqrt(var_Y*n)/n = sqrt(var_Y)*sqrt(n)/n = sqrt(var_Y)/sqrt(n) = std_Y/sqrt(n);
    


if length(fig)<5 || (length(fig)==5 && fig(5)==0)
    hold off;
else
    hold on;
end

if barNOTplot
    bar(x,mean_Y,'FaceColor',[.6 .6 .6],'EdgeColor','none'); hold on;
    ls='none';
else
    ls='-';
end

lw=2;

errorbar(x,mean_Y,std_Y,'Color',col,'LineWidth',lw,'LineStyle',ls);
title(titleString);



