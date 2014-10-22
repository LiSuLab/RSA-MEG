function stringORstringInCell=deunderscore(stringORstringInCell)

% replace underscores
if iscell(stringORstringInCell)
    for lineI=1:numel(stringORstringInCell)
        line=stringORstringInCell{lineI};
        line(line==95)='-';
        stringORstringInCell{lineI}=line;
    end
else
    stringORstringInCell(stringORstringInCell==95)='-';
end  