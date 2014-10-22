function exportCurrentFigAsPDF(filespec,appendFlag,userOptions)
% exports the current figures to the file [filespec,'.ps'] in postscript
% format. if appendFlag is 0 any existing file [filespec,'.ps'] is
% overwritten. if appendFlag is 1 the figure is appended to the existing
% postscript file [filespec,'.ps']. if appendFlag is 3 (default) the figure
% is appended to the existing postscript file [filespec,'.ps'] and exported
% to a separate file [filespec,'_',num2str(gcf),'.ps'].

if ~exist('appendFlag','var'), appendFlag=0; end;
if ~isfield(userOptions,'dpi'), userOptions.dpi = 300; end;
if ~isfield(userOptions,'tightInset'), userOptions.tightInset = false; end;

% If we are trimming the paper size to match the figure
if userOptions.tightInset
    setPapertoFigPos;
end

switch appendFlag
    case 0
        print('-dpdf',sprintf('-r%d',userOptions.dpi),filespec);
    case 1
        print('-dpdf',sprintf('-r%d',userOptions.dpi),'-append',filespec);
    case 3
        print('-dpdf',sprintf('-r%d',userOptions.dpi),[filespec,'_',num2str(gcf)]);
        print('-dpdf',sprintf('-r%d',userOptions.dpi),'-append',filespec);
end