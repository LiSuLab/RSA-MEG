function out = partial_compare(R, target_model, control_models, varargin)
%PARTIAL_COMPARE Static partial model-to-neural RDM similarity controlling for covariate RDMs.
%   out = rsa_meeg.mapping.partial_compare(R, target_model, control_models, ...)
%
%   Computes partial Pearson or partial Spearman (rank-based) RDM similarity
%   between neural RDMs R and one or more target model RDMs while partialling
%   out one or more control/nuisance model RDMs at each center and window.
%
%   Inputs:
%     R              - Neural RDM struct from rsa_meeg.rdm.searchlight or
%                      rsa_meeg.rdm.average.
%     target_model   - Target model RDM ([n_pairs x n_targets] matrix or struct
%                      with condition_ids and values).
%     control_models - Control/covariate model RDM(s) ([n_pairs x n_controls]
%                      matrix, struct with condition_ids and values, or cell
%                      array of model structs/matrices).
%
%   Name-Value Options:
%     'Correlation'  - 'spearman' (default) or 'pearson'.
%
%   Output:
%     out - Struct with fields:
%             .values        [n_centers x n_windows x n_targets] partial correlations
%             .center_ids    center identifiers from R
%             .windows       window metadata from R
%             .condition_ids aligned condition identifiers
%             .correlation   'spearman' or 'pearson'
%             .n_controls    number of controlled covariate RDM columns
%             .n_pairs       [n_centers x n_windows] finite pair counts
%             .assumption    descriptive interpretation note

p = inputParser;
p.FunctionName = 'rsa_meeg.mapping.partial_compare';
addParameter(p, 'Correlation', 'spearman');
parse(p, varargin{:});
corr_type = lower(char(p.Results.Correlation));
if ~ismember(corr_type, {'spearman', 'pearson'})
    error('rsa_meeg:mapping:PartialCorrelation', 'Correlation for partial_compare must be spearman or pearson.');
end

[m_target, ids] = rsa_meeg.mapping.prepare_model(R, target_model);
if ndims(m_target) > 2 || size(m_target, 1) ~= size(R.values, 1)
    error('rsa_meeg:mapping:StaticShape', 'Target model must be pairs x models.');
end

m_ctrl = prepare_control_matrix(R, control_models);
if size(m_ctrl, 2) < 1
    error('rsa_meeg:mapping:EmptyControl', 'At least one control model column is required.');
end

nc = size(R.values, 2);
nt = size(R.values, 3);
nm = size(m_target, 2);
n_ctrl = size(m_ctrl, 2);

v = nan(nc, nt, nm);
n_pairs = zeros(nc, nt);

for t = 1:nt
    for c = 1:nc
        y = R.values(:, c, t);
        keep = isfinite(y) & all(isfinite(m_target), 2) & all(isfinite(m_ctrl), 2);
        n_keep = sum(keep);
        n_pairs(c, t) = n_keep;
        if n_keep == 0
            continue
        end
        if n_keep < n_ctrl + 3
            error('rsa_meeg:mapping:PartialSamples', ...
                'Too few finite RDM pairs (%d) to partial out %d control model(s).', n_keep, n_ctrl);
        end

        yk = y(keep);
        Xk = m_target(keep, :);
        Zk = m_ctrl(keep, :);

        if strcmp(corr_type, 'spearman')
            yk = local_ranks(yk);
            for j = 1:nm
                Xk(:, j) = local_ranks(Xk(:, j));
            end
            for k = 1:n_ctrl
                Zk(:, k) = local_ranks(Zk(:, k));
            end
        end

        sd_z = std(Zk, 0, 1);
        if any(sd_z == 0)
            error('rsa_meeg:mapping:PartialRank', 'Control model contains a constant predictor column.');
        end
        Zk_std = bsxfun(@rdivide, bsxfun(@minus, Zk, mean(Zk, 1)), sd_z);
        C_des = [ones(n_keep, 1), Zk_std];
        if rank(C_des) < n_ctrl + 1
            error('rsa_meeg:mapping:PartialRank', 'Control model design matrix is rank deficient.');
        end

        [Q, ~] = qr(C_des, 0);
        ry = yk - Q * (Q.' * yk);
        norm_ry = sqrt(sum(ry.^2));
        scale_y = sqrt(sum((yk - mean(yk)).^2));

        for j = 1:nm
            xj = Xk(:, j);
            rx = xj - Q * (Q.' * xj);
            norm_rx = sqrt(sum(rx.^2));
            scale_x = sqrt(sum((xj - mean(xj)).^2));
            if norm_ry <= 1e-12 * max(1, scale_y) || norm_rx <= 1e-12 * max(1, scale_x)
                v(c, t, j) = NaN;
            else
                r_val = sum(ry .* rx) / (norm_ry * norm_rx);
                v(c, t, j) = max(-1.0, min(1.0, r_val));
            end
        end
    end
end

out = struct();
out.values = v;
out.center_ids = R.center_ids;
out.windows = R.windows;
out.condition_ids = ids;
out.correlation = corr_type;
out.n_controls = n_ctrl;
out.n_pairs = n_pairs;
out.assumption = 'Partial correlation across dependent RDM edges controlling for covariate models; values are descriptive, with no analytic edgewise p-value.';
end

function m_ctrl = prepare_control_matrix(R, control_models)
if iscell(control_models)
    if isempty(control_models)
        error('rsa_meeg:mapping:EmptyControl', 'control_models cell array cannot be empty.');
    end
    blocks = cell(1, numel(control_models));
    for k = 1:numel(control_models)
        bk = rsa_meeg.mapping.prepare_model(R, control_models{k});
        if ndims(bk) > 2
            error('rsa_meeg:mapping:StaticShape', 'Each control model must be pairs x models.');
        end
        blocks{k} = bk;
    end
    m_ctrl = horzcat(blocks{:});
else
    m_ctrl = rsa_meeg.mapping.prepare_model(R, control_models);
    if ndims(m_ctrl) > 2
        error('rsa_meeg:mapping:StaticShape', 'Control models must be pairs x models.');
    end
end
end

function z = local_ranks(x)
[s, ord] = sort(x(:));
z = zeros(size(x(:)));
i = 1;
n = numel(s);
while i <= n
    j = i;
    while j < n && s(j + 1) == s(i)
        j = j + 1;
    end
    z(ord(i:j)) = (i + j) / 2;
    i = j + 1;
end
z = reshape(z, size(x));
end
