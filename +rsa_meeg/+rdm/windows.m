function w=windows(sample_times_s,width_samples,step_samples)
%WINDOWS Full inclusive sample windows; no padding or endpoint loss.
% Corrects fencepost behavior in LiSuLab +rsa/+meg/getSlidingWindowSpec.m,
% commit c0dfd32f809eb816c4e94ae765feab0e4d12ab5a.
if ~isnumeric(sample_times_s) || isempty(sample_times_s) || ~isreal(sample_times_s) || any(~isfinite(sample_times_s(:))) || any(diff(sample_times_s(:))<=0)
    error('rsa_meeg:rdm:Times','Sample times must be finite and strictly increasing.');
end
t=numel(sample_times_s);
if ~isscalar(width_samples) || ~isnumeric(width_samples) || width_samples<1 || width_samples>t || width_samples~=floor(width_samples)
    error('rsa_meeg:rdm:Width','Width must be an integer between 1 and number of samples.');
end
if ~isscalar(step_samples) || ~isnumeric(step_samples) || step_samples<1 || step_samples~=floor(step_samples)
    error('rsa_meeg:rdm:Step','Step must be a positive integer.');
end
starts=1:step_samples:(t-width_samples+1);
ends=starts+width_samples-1;
tm=double(sample_times_s(:).');
w=struct('indices',{arrayfun(@(a,b) a:b,starts,ends,'UniformOutput',false)}, ...
    'start_s',tm(starts),'center_s',(tm(starts)+tm(ends))/2, ...
    'end_s',tm(ends),'width_samples',width_samples,'step_samples',step_samples);
end
