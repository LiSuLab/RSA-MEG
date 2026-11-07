function [P, info] = shrinkage_precision(residuals_or_data, varargin)
%SHRINKAGE_PRECISION Ledoit-Wolf, OAS, or fixed-shrinkage spatial/spatiotemporal precision estimator.
%   [P, info] = rsa_meeg.rdm.shrinkage_precision(residuals_or_data, ...)
%   [P, info] = rsa_meeg.rdm.shrinkage_precision(ds, neighborhoods, ...)
%
%   Estimates positive-definite noise precision matrices P = inv(Sigma_shrunk)
%   for cross-validated Mahalanobis (crossnobis) RDM computation, suppressing
%   EEG volume-conduction common-mode crosstalk and regularizing high-dimensional
%   spatiotemporal searchlight patterns.
%
%   The returned P (either a [D x D] matrix or a 1 x n_centers cell array of
%   center-specific precision matrices) can be passed directly into:
%     rsa_meeg.rdm.searchlight(ds, hood, 'Metric', 'crossnobis', 'Precision', P, ...)
%
%   Inputs:
%     residuals_or_data - Either:
%                         (1) An rsa_meeg.dataset struct ds;
%                         (2) A 2-D [n_samples x D] numeric residual matrix; or
%                         (3) A 3-D [n_obs x n_features x n_times] numeric array.
%     neighborhoods     - Optional cell array of local feature indices (may be
%                         passed as the second positional argument or via the
%                         'Neighborhoods' name-value parameter).
%
%   Name-Value Options:
%     'Neighborhoods'          - Cell array of local feature index vectors.
%     'Method'                 - 'ledoit-wolf' (default), 'oas', 'fixed', or 'diagonal'.
%     'Shrinkage'              - Optional scalar shrinkage intensity lambda in [0, 1].
%                                When supplied, overrides Method to 'fixed'.
%     'Mode'                   - 'spatiotemporal' (default), 'spatial', or 'temporal',
%                                matching rsa_meeg.rdm.searchlight pattern mode.
%     'WindowSamples'          - Positive integer window width in samples (default 1).
%                                Used to match spatiotemporal pattern dimensionality
%                                when Mode is 'spatiotemporal' or 'temporal'.
%     'FeatureTypes'           - Optional feature type filter (when input is a dataset).
%     'ConditionIds'           - Optional condition label vector for a raw numeric array.
%     'SubtractConditionMeans' - Logical scalar (default true). Subtracts condition
%                                (or condition x fold) mean patterns before estimating
%                                residual noise covariance.
%     'BaselineWindow'         - Optional [t_start_s, t_end_s] time interval (in s)
%                                restricting noise estimation to baseline samples.
%
%   Outputs:
%     P    - [D x D] symmetric positive-definite precision matrix, or 1 x n_centers
%            cell array of symmetric positive-definite precision matrices.
%     info - Struct with fields method, shrinkage, condition_number, pattern_dimensions,
%            and n_samples.

hood_arg = [];
if ~isempty(varargin) && iscell(varargin{1}) && ...
        (isempty(varargin{1}) || isnumeric(varargin{1}{1}))
    hood_arg = varargin{1};
    varargin = varargin(2:end);
end

p = inputParser;
p.FunctionName = 'rsa_meeg.rdm.shrinkage_precision';
addParameter(p, 'Neighborhoods', []);
addParameter(p, 'Method', 'ledoit-wolf');
addParameter(p, 'Shrinkage', []);
addParameter(p, 'Mode', 'spatiotemporal');
addParameter(p, 'WindowSamples', 1);
addParameter(p, 'FeatureTypes', []);
addParameter(p, 'ConditionIds', []);
addParameter(p, 'SubtractConditionMeans', true);
addParameter(p, 'BaselineWindow', []);
parse(p, varargin{:});
o = p.Results;

neighborhoods = hood_arg;
if ~isempty(o.Neighborhoods)
    neighborhoods = o.Neighborhoods;
end

method = lower(string(o.Method));
if ~isempty(o.Shrinkage)
    if ~isscalar(o.Shrinkage) || ~isnumeric(o.Shrinkage) || ~isreal(o.Shrinkage) || ...
            ~isfinite(o.Shrinkage) || o.Shrinkage < 0 || o.Shrinkage > 1
        error('rsa_meeg:rdm:Shrinkage', 'Shrinkage must be a finite scalar in [0, 1].');
    end
    method = "fixed";
end
if ~isscalar(method) || ~any(method == ["ledoit-wolf", "lw", "oas", "fixed", "diagonal"])
    error('rsa_meeg:rdm:ShrinkageMethod', 'Method must be ledoit-wolf, oas, fixed, or diagonal.');
end
if method == "fixed" && isempty(o.Shrinkage)
    error('rsa_meeg:rdm:Shrinkage', 'Method="fixed" requires a Shrinkage value in [0, 1].');
end

mode = lower(string(o.Mode));
if ~isscalar(mode) || ~any(mode == ["spatial", "temporal", "spatiotemporal"])
    error('rsa_meeg:rdm:Mode', 'Mode must be spatial, temporal, or spatiotemporal.');
end
win_samples = double(o.WindowSamples);
if ~isscalar(win_samples) || ~isnumeric(win_samples) || ~isreal(win_samples) || ...
        ~isfinite(win_samples) || win_samples < 1 || win_samples ~= floor(win_samples)
    error('rsa_meeg:rdm:WindowSamples', 'WindowSamples must be a positive integer scalar.');
end

% Extract data array, condition labels, and bad features
is_ds = isstruct(residuals_or_data);
if is_ds
    rsa_meeg.validate_dataset(residuals_or_data);
    raw_data = double(residuals_or_data.data);
    sample_times_s = residuals_or_data.sample_times_s;
    cond_ids = residuals_or_data.condition_ids;
    cond_order = residuals_or_data.condition_order;
    fold_ids = residuals_or_data.fold_ids;
    bad_features = residuals_or_data.bad_features;
    feat_types = string(residuals_or_data.feature_types);
else
    if ~isnumeric(residuals_or_data) || ~isreal(residuals_or_data) || isempty(residuals_or_data) || ...
            ndims(residuals_or_data) > 3 || any(~isfinite(residuals_or_data(:)))
        error('rsa_meeg:rdm:InputData', 'Input must be a dataset struct or finite real 2-D/3-D numeric array.');
    end
    raw_data = double(residuals_or_data);
    sample_times_s = 1:size(raw_data, 3);
    cond_ids = o.ConditionIds;
    if ~isempty(cond_ids)
        if numel(cond_ids) ~= size(raw_data, 1)
            error('rsa_meeg:rdm:ConditionIds', 'ConditionIds length must match size(data, 1).');
        end
        cond_order = unique(cond_ids, 'stable');
    else
        cond_order = [];
    end
    fold_ids = [];
    bad_features = false(1, size(raw_data, 2));
    feat_types = repmat("unknown", 1, size(raw_data, 2));
end

% Optional baseline time window restriction
if ~isempty(o.BaselineWindow)
    bw = double(o.BaselineWindow);
    if numel(bw) ~= 2 || any(~isfinite(bw)) || bw(1) > bw(2)
        error('rsa_meeg:rdm:BaselineWindow', 'BaselineWindow must be [t_start_s, t_end_s].');
    end
    t_keep = sample_times_s >= bw(1) & sample_times_s <= bw(2);
    if ~any(t_keep)
        error('rsa_meeg:rdm:BaselineWindow', 'No sample times fall inside BaselineWindow.');
    end
    raw_data = raw_data(:, :, t_keep);
end

% Subtract condition means to isolate within-condition residual noise
if logical(o.SubtractConditionMeans) && ~isempty(cond_order)
    [~, cond_idx] = ismember(cond_ids, cond_order);
    nc = numel(cond_order);
    if ~isempty(fold_ids)
        [~, ~, fold_idx] = unique(fold_ids, 'stable');
        nfold = max(fold_idx);
        % If multiple observations exist per condition x fold cell, subtract cell mean;
        % otherwise subtract overall condition mean across folds.
        cell_counts = accumarray([cond_idx(:), fold_idx(:)], 1, [nc, nfold]);
        if all(cell_counts(:) >= 2)
            for fi = 1:nfold
                for ci = 1:nc
                    mask = (fold_idx(:) == fi) & (cond_idx(:) == ci);
                    raw_data(mask, :, :) = bsxfun(@minus, raw_data(mask, :, :), mean(raw_data(mask, :, :), 1));
                end
            end
        else
            for ci = 1:nc
                mask = (cond_idx(:) == ci);
                raw_data(mask, :, :) = bsxfun(@minus, raw_data(mask, :, :), mean(raw_data(mask, :, :), 1));
            end
        end
    else
        for ci = 1:nc
            mask = (cond_idx(:) == ci);
            raw_data(mask, :, :) = bsxfun(@minus, raw_data(mask, :, :), mean(raw_data(mask, :, :), 1));
        end
    end
end

if ~isempty(o.FeatureTypes)
    selected_types = string(o.FeatureTypes);
else
    selected_types = [];
end

% Case A: Single matrix estimation (no neighborhoods supplied)
if isempty(neighborhoods)
    good_idx = find(~bad_features);
    if ~isempty(selected_types)
        good_idx = good_idx(ismember(feat_types(good_idx), selected_types));
    end
    if isempty(good_idx)
        error('rsa_meeg:rdm:EmptyFeatures', 'No valid features remain for precision estimation.');
    end
    X_pat = extract_residual_patterns(raw_data(:, good_idx, :), mode, win_samples);
    [P, lam, cond_num] = estimate_single_precision(X_pat, method, o.Shrinkage);
    info = struct('method', char(method), 'shrinkage', lam, ...
        'condition_number', cond_num, 'pattern_dimensions', size(P, 1), ...
        'n_samples', size(X_pat, 1));
    return
end

% Case B: Per-center searchlight neighborhood cell estimation
if ~iscell(neighborhoods)
    error('rsa_meeg:rdm:Neighborhoods', 'Neighborhoods must be a cell array of local feature indices.');
end
ncenter = numel(neighborhoods);
nf = size(raw_data, 2);
P = cell(1, ncenter);
lam_vec = zeros(1, ncenter);
cond_vec = zeros(1, ncenter);
dim_vec = zeros(1, ncenter);
nsamp_vec = zeros(1, ncenter);

for c = 1:ncenter
    idx = neighborhoods{c};
    if ~isnumeric(idx) || ~isreal(idx) || ~isvector(idx) || isempty(idx) || ...
            any(~isfinite(idx)) || any(idx < 1 | idx > nf | idx ~= floor(idx)) || numel(unique(idx)) ~= numel(idx)
        error('rsa_meeg:rdm:NeighborhoodIndex', 'Each neighborhood needs unique local feature indices.');
    end
    idx = idx(:).';
    idx = idx(~bad_features(idx));
    if ~isempty(selected_types)
        idx = idx(ismember(feat_types(idx), selected_types));
    end
    if isempty(idx)
        error('rsa_meeg:rdm:EmptyNeighborhood', 'No good features remain in neighborhood %d.', c);
    end
    X_pat = extract_residual_patterns(raw_data(:, idx, :), mode, win_samples);
    [P{c}, lam_vec(c), cond_vec(c)] = estimate_single_precision(X_pat, method, o.Shrinkage);
    dim_vec(c) = size(P{c}, 1);
    nsamp_vec(c) = size(X_pat, 1);
end

info = struct('method', char(method), 'shrinkage', lam_vec, ...
    'condition_number', cond_vec, 'pattern_dimensions', dim_vec, ...
    'n_samples', nsamp_vec);
end

function X_pat = extract_residual_patterns(sub_data, mode, win_samples)
[n_obs, n_feat, n_time] = size(sub_data);
if n_time == 1 && win_samples == 1
    X_pat = reshape(sub_data, [n_obs, n_feat]);
    return
end
switch mode
    case "spatial"
        % In spatial mode, searchlight averages across window samples;
        % pool all available windows or time samples to estimate spatial covariance.
        if n_time >= win_samples && win_samples > 1
            n_win = n_time - win_samples + 1;
            X_pat = zeros(n_obs * n_win, n_feat);
            for w = 1:n_win
                rows = (w - 1) * n_obs + (1:n_obs);
                X_pat(rows, :) = reshape(mean(sub_data(:, :, w:(w + win_samples - 1)), 3), [n_obs, n_feat]);
            end
        else
            X_pat = reshape(permute(sub_data, [1, 3, 2]), [n_obs * n_time, n_feat]);
        end
    case "temporal"
        if n_time < win_samples
            error('rsa_meeg:rdm:WindowSamples', 'Time axis (%d) is shorter than WindowSamples (%d).', n_time, win_samples);
        end
        feat_avg = reshape(mean(sub_data, 2), [n_obs, n_time]);
        n_win = n_time - win_samples + 1;
        X_pat = zeros(n_obs * n_win, win_samples);
        for w = 1:n_win
            rows = (w - 1) * n_obs + (1:n_obs);
            X_pat(rows, :) = feat_avg(:, w:(w + win_samples - 1));
        end
    otherwise % spatiotemporal
        if win_samples == 1
            X_pat = reshape(permute(sub_data, [1, 3, 2]), [n_obs * n_time, n_feat]);
        else
            if n_time < win_samples
                error('rsa_meeg:rdm:WindowSamples', 'Time axis (%d) is shorter than WindowSamples (%d).', n_time, win_samples);
            end
            n_win = n_time - win_samples + 1;
            ndim = n_feat * win_samples;
            X_pat = zeros(n_obs * n_win, ndim);
            for w = 1:n_win
                rows = (w - 1) * n_obs + (1:n_obs);
                X_pat(rows, :) = reshape(sub_data(:, :, w:(w + win_samples - 1)), [n_obs, ndim]);
            end
        end
end
end

function [P_mat, lam, cond_num] = estimate_single_precision(X, method, fixed_lambda)
[N, D] = size(X);
if N < 2
    error('rsa_meeg:rdm:Samples', 'At least 2 residual samples are required to estimate noise precision.');
end
X = bsxfun(@minus, X, mean(X, 1));
S = (X.' * X) / N;
S = 0.5 * (S + S.');
tr_S = trace(S);
mu = tr_S / D;
if mu <= 0
    error('rsa_meeg:rdm:ZeroVariance', 'Residual data have zero total variance.');
end

switch method
    case {"ledoit-wolf", "lw"}
        delta2 = sum(sum((S - mu * eye(D)).^2)) / D;
        if delta2 <= eps(mu^2)
            lam = 1.0;
        else
            row_sq = sum(X.^2, 2);
            term1 = sum(row_sq.^2) / (N^2 * D);
            term2 = sum(S(:).^2) / (N * D);
            beta2 = max(0, term1 - term2);
            lam = min(1.0, max(0.0, beta2 / delta2));
        end
        Sigma = (1 - lam) * S + lam * mu * eye(D);

    case "oas"
        tr_S2 = sum(S(:).^2);
        denom_factor = tr_S2 - (tr_S^2) / D;
        if denom_factor <= eps(mu^2)
            lam = 1.0;
        else
            num = (1 - 2 / D) * tr_S2 + tr_S^2;
            den = (N + 1 - 2 / D) * denom_factor;
            lam = min(1.0, max(0.0, num / den));
        end
        Sigma = (1 - lam) * S + lam * mu * eye(D);

    case "fixed"
        lam = double(fixed_lambda);
        Sigma = (1 - lam) * S + lam * mu * eye(D);

    case "diagonal"
        lam = 1.0;
        d_vars = diag(S);
        min_pos = max(eps(mu), 1e-6 * mu);
        d_vars = max(d_vars, min_pos);
        Sigma = diag(d_vars);
end

Sigma = 0.5 * (Sigma + Sigma.');
[V, D_eig] = eig(Sigma);
evals = real(diag(D_eig));
floor_val = max(1e-10 * max(evals), eps(max(evals)));
evals = max(evals, floor_val);
P_mat = real(V) * diag(1 ./ evals) * real(V).';
P_mat = 0.5 * (P_mat + P_mat.');
cond_num = max(evals) / min(evals);
end
