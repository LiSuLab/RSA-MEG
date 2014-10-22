function Y=temporallySmoothTimeSpaceMatrix(Y,FWHM)
% Y=temporallySmoothTimeSpaceMatrix(Y,FWHM)
% temporally smooths the time-by-space matrix Y with a gaussian kernel of
% full width at half maximum FWHM

%% define smoothing kernel
kernel=gaussian_nk(FWHM)';


%% smooth temporally
Y=conv2(Y,kernel,'same');