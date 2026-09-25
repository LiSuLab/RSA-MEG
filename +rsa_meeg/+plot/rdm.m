function ax=rdm(R,center,window,varargin)
%RDM Display one RDM with explicit condition labels. Undefined entries NaN.
if nargin<2,center=1;end
if nargin<3,window=1;end
n=numel(R.condition_ids);m=zeros(n);v=R.values(:,center,window);
for k=1:size(R.pairs,1),i=R.pairs(k,1);j=R.pairs(k,2);m(i,j)=v(k);m(j,i)=v(k);end
if isempty(varargin),ax=axes;else,ax=varargin{1};end
imagesc(ax,m);axis(ax,'image');colorbar(ax);
set(ax,'XTick',1:n,'YTick',1:n,'XTickLabel',string(R.condition_ids),'YTickLabel',string(R.condition_ids));
title(ax,sprintf('%s RDM, %.3f s',R.metric,R.windows.center_s(window)),'Interpreter','none');
end
