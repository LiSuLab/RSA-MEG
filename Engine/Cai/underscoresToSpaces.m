%  underscoresToSpaces is a function based on Niko Kriegeskorte's deunderscore
%  function.  It takes an incomming string and replaces all underscores (which
%  are sometimes interpreted by MATLAB as subscript indicators in figures) as
%  spaces (which aren't).
%
%  Cai Wingfield 11-2009

function stringORstringInCell=underscoresToSpaces(stringORstringInCell)

if iscell(stringORstringInCell)
    for lineI=1:numel(stringORstringInCell)
        line=stringORstringInCell{lineI};
        line(line==95)=' ';
        stringORstringInCell{lineI}=line;
    end
else
    stringORstringInCell(stringORstringInCell==95)=' ';
end  