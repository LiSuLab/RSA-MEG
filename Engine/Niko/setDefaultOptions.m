function options=setDefaultOptions(passedOptions,defaultOptions)
% this function returns the structure options, which contains all fields
% and field contents from the structure passedOptions and also any
% additional fields present in defaultOptions. the additional fields
% present in defaultOptions, but not in passedOptions are set to the values
% they have in defaultOptions.
% this is useful in functions that take an options structure as argument.
% if such a function automatically sets default options, the caller of the
% function is spared the toil of defining all options.

options=passedOptions;

fn=fieldnames(defaultOptions);
for fieldI=1:numel(fn)
    if ~isfield(options,fn{fieldI})
        options.(fn{fieldI})=defaultOptions.(fn{fieldI});
    end
end



