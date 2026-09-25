function ax=map(result,model,varargin)
%MAP Center x window similarity heatmap; each row is an analysis center.
% Uses actual window centers, no fabricated sample-rate/time origin.
if nargin<2,model=1;end
if isempty(varargin),ax=axes;else,ax=varargin{1};end
times=result.windows.center_s;
imagesc(ax,1:numel(times),1:numel(result.center_ids),result.values(:,:,model));
ticks=unique(round(linspace(1,numel(times),min(7,numel(times)))));
set(ax,'XTick',ticks,'XTickLabel',compose('%.3g',times(ticks)));
xlabel(ax,'Window center (s)');ylabel(ax,'Analysis center');colorbar(ax);
title(ax,sprintf('Model %d similarity (%s)',model,result.correlation),'Interpreter','none');
end
