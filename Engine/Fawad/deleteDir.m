% FJ 10-2014

function deleteDir(userOptions)

    if (~userOptions.keepTMaps)
        dirToDel= fullfile(userOptions.rootPath, '/Maps');
        rmdir(dirToDel, 's');
    end
    
    if (~userOptions.keepImageData)
        dirToDel= fullfile(userOptions.rootPath, '/ImageData');
        rmdir(dirToDel, 's');
    end
end
