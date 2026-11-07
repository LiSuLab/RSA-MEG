function [B, cov_true, rdm_true] = generate_beta_patterns(cluster_spec, n_features, varargin)
%GENERATE_BETA_PATTERNS Recursive hierarchical condition covariance and multivariate pattern generator.
%   [B, cov_true, rdm_true] = rsa_meeg.sim.generate_beta_patterns(cluster_spec, n_features, ...)
%
%   Reimplements and generalizes the hierarchical cluster pattern simulation
%   framework of Li Su and Cai Wingfield (LiSuLab / rsatoolbox_matlab
%   generateBetaPatterns). Supports arbitrary nested cell-tree specifications
%   such as {0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}} as well as explicit
%   condition covariance matrices.
%
%   Inputs:
%     cluster_spec - Nested cell array specifying the hierarchical condition
%                    tree, or a symmetric positive-definite [K x K] numeric
%                    condition covariance matrix.
%                    In cell-tree format:
%                      * Leaf node: {spread_within, n_conditions} (e.g., {0.3, 5})
%                      * Internal node: {spread_between, child_1, child_2, ...}
%                        (e.g., {0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}})
%     n_features   - Positive integer number of spatial/spatiotemporal features (P).
%
%   Name-Value Options:
%     'Parameterization'  - 'auto' (default), 'spread', or 'correlation'.
%                           In 'spread' mode, outer parameters specify cluster
%                           centroid variance/separation and leaf parameters
%                           specify within-cluster exemplar variance. In
%                           'correlation' mode, leaf parameters specify
%                           within-cluster correlation and parent parameters
%                           specify between-cluster correlation. 'auto' selects
%                           'spread' when root parameter >= leaf parameter and
%                           'correlation' otherwise.
%     'NormalizeDiagonal' - Logical scalar (default true). Scales cov_true so
%                           that diag(cov_true) == 1.
%     'ExactCovariance'   - Logical scalar (default false). When true and
%                           n_features >= K, orthonormalizes the Gaussian draw
%                           so (B * B') / n_features equals cov_true to machine
%                           precision. When false, draws independent standard
%                           normal features so E[(B * B') / n_features] = cov_true.
%     'RDMMetric'         - 'sqeuclidean' (default) or 'correlation' for rdm_true.
%     'Scale'             - Positive scalar amplitude multiplier (default 1.0).
%     'Seed'              - Optional nonnegative integer RNG seed.
%
%   Outputs:
%     B        - [K x n_features] matrix of true multivariate condition patterns.
%     cov_true - [K x K] symmetric positive-definite true condition covariance.
%     rdm_true - Struct compatible with rsa_meeg.mapping.compare and
%                rsa_meeg.mapping.partial_compare containing:
%                  .values        [n_pairs x 1] upper-triangular RDM vector
%                  .condition_ids [1 x K] condition indices (1:K)
%                  .pairs         [n_pairs x 2] lexicographic pair indices
%                  .square        [K x K] symmetric hollow RDM matrix
%                  .metric        char ('sqeuclidean' or 'correlation')

if ~isscalar(n_features) || ~isnumeric(n_features) || ~isreal(n_features) || ...
        ~isfinite(n_features) || n_features < 1 || n_features ~= floor(n_features)
    error('rsa_meeg:sim:Features', 'n_features must be a positive integer scalar.');
end
n_features = double(n_features);

p = inputParser;
p.FunctionName = 'rsa_meeg.sim.generate_beta_patterns';
addParameter(p, 'Parameterization', 'auto');
addParameter(p, 'NormalizeDiagonal', true);
addParameter(p, 'ExactCovariance', false);
addParameter(p, 'RDMMetric', 'sqeuclidean');
addParameter(p, 'Scale', 1.0);
addParameter(p, 'Seed', []);
parse(p, varargin{:});
o = p.Results;

param_mode = lower(string(o.Parameterization));
if ~isscalar(param_mode) || ~any(param_mode == ["auto", "spread", "correlation"])
    error('rsa_meeg:sim:Parameterization', 'Parameterization must be auto, spread, or correlation.');
end
if ~(islogical(o.NormalizeDiagonal) || isnumeric(o.NormalizeDiagonal)) || ~isscalar(o.NormalizeDiagonal)
    error('rsa_meeg:sim:NormalizeDiagonal', 'NormalizeDiagonal must be a logical scalar.');
end
if ~(islogical(o.ExactCovariance) || isnumeric(o.ExactCovariance)) || ~isscalar(o.ExactCovariance)
    error('rsa_meeg:sim:ExactCovariance', 'ExactCovariance must be a logical scalar.');
end
rdm_metric = lower(string(o.RDMMetric));
if ~isscalar(rdm_metric) || ~any(rdm_metric == ["sqeuclidean", "correlation"])
    error('rsa_meeg:sim:RDMMetric', 'RDMMetric must be sqeuclidean or correlation.');
end
if ~isscalar(o.Scale) || ~isnumeric(o.Scale) || ~isreal(o.Scale) || ~isfinite(o.Scale) || o.Scale <= 0
    error('rsa_meeg:sim:Scale', 'Scale must be a positive finite scalar.');
end
if ~isempty(o.Seed)
    if ~isscalar(o.Seed) || ~isnumeric(o.Seed) || ~isreal(o.Seed) || ~isfinite(o.Seed) || o.Seed < 0 || o.Seed ~= floor(o.Seed)
        error('rsa_meeg:sim:Seed', 'Seed must be a nonnegative integer scalar.');
    end
    prior_rng = rng;
    cleanup_rng = onCleanup(@() rng(prior_rng));
    rng(double(o.Seed), 'twister');
end

if isnumeric(cluster_spec)
    if ~isreal(cluster_spec) || ndims(cluster_spec) > 2 || size(cluster_spec, 1) ~= size(cluster_spec, 2) || ...
            size(cluster_spec, 1) < 2 || any(~isfinite(cluster_spec(:)))
        error('rsa_meeg:sim:ClusterSpec', 'Numeric cluster_spec must be a finite K x K matrix with K >= 2.');
    end
    if norm(cluster_spec - cluster_spec.', 'fro') > 1e-10 * max(1, norm(cluster_spec, 'fro'))
        error('rsa_meeg:sim:ClusterSpec', 'Numeric cluster_spec must be symmetric.');
    end
    cov_true = 0.5 * (double(cluster_spec) + double(cluster_spec).');
else
    if ~iscell(cluster_spec) || isempty(cluster_spec)
        error('rsa_meeg:sim:ClusterSpec', 'cluster_spec must be a nonempty nested cell array or numeric matrix.');
    end
    resolved_mode = resolve_tree_mode(cluster_spec, param_mode);
    if resolved_mode == "spread"
        cov_true = build_spread_covariance(cluster_spec, 0.0);
    else
        cov_true = build_correlation_covariance(cluster_spec);
    end
end

nc = size(cov_true, 1);
if nc < 2
    error('rsa_meeg:sim:Conditions', 'At least two conditions are required.');
end

if logical(o.NormalizeDiagonal)
    d = sqrt(diag(cov_true));
    if any(d <= 0)
        error('rsa_meeg:sim:Covariance', 'cov_true must have strictly positive diagonal variances.');
    end
    cov_true = cov_true ./ (d * d.');
    cov_true(1:nc+1:end) = 1.0;
end

cov_true = (double(o.Scale)^2) * 0.5 * (cov_true + cov_true.');
[V_eig, D_eig] = eig(cov_true);
evals = real(diag(D_eig));
if min(evals) <= -1e-10 * max(1, max(evals))
    error('rsa_meeg:sim:CovarianceNotPSD', 'Constructed condition covariance matrix is not positive semidefinite.');
end
evals = max(evals, 0);

% Matrix square root factor L such that L * L' == cov_true
L = real(V_eig) * diag(sqrt(evals)) * real(V_eig).';

if logical(o.ExactCovariance)
    if n_features < nc
        error('rsa_meeg:sim:ExactCovarianceDim', ...
            'ExactCovariance=true requires n_features (%d) >= n_conditions (%d).', n_features, nc);
    end
    Z = randn(n_features, nc);
    [Q, R_qr] = qr(Z, 0);
    signs = sign(diag(R_qr));
    signs(signs == 0) = 1;
    Q = Q * diag(signs);
    % Q' * Q == I_nc, so (Q' * sqrt(n_features)) has exact sample covariance I_nc
    Z_white = sqrt(n_features) * Q.';
else
    Z_white = randn(nc, n_features);
end

B = L * Z_white;

% Construct analytical ground-truth RDM
pairs = nchoosek(1:nc, 2);
np = size(pairs, 1);
d_diag = diag(cov_true);
if rdm_metric == "sqeuclidean"
    sq_mat = bsxfun(@plus, d_diag, d_diag.') - 2 * cov_true;
    sq_mat = max(0, 0.5 * (sq_mat + sq_mat.'));
    sq_mat(1:nc+1:end) = 0;
else
    sd = sqrt(max(d_diag, eps));
    corr_mat = cov_true ./ (sd * sd.');
    sq_mat = max(0, 1 - 0.5 * (corr_mat + corr_mat.'));
    sq_mat(1:nc+1:end) = 0;
end

rdm_vals = zeros(np, 1);
for k = 1:np
    rdm_vals(k) = sq_mat(pairs(k, 1), pairs(k, 2));
end

rdm_true = struct();
rdm_true.values = rdm_vals;
rdm_true.condition_ids = 1:nc;
rdm_true.pairs = pairs;
rdm_true.square = sq_mat;
rdm_true.metric = char(rdm_metric);
end

function mode_out = resolve_tree_mode(spec, param_mode)
validate_tree_node(spec);
if param_mode ~= "auto"
    mode_out = param_mode;
    return
end
if is_leaf_node(spec)
    mode_out = "spread";
    return
end
root_val = double(spec{1});
leaf_vals = collect_leaf_values(spec);
if root_val >= mean(leaf_vals)
    mode_out = "spread";
else
    mode_out = "correlation";
end
end

function validate_tree_node(node)
if ~iscell(node) || numel(node) < 2
    error('rsa_meeg:sim:ClusterSpec', 'Each tree node must be a cell array with at least 2 elements.');
end
val = node{1};
if ~isscalar(val) || ~isnumeric(val) || ~isreal(val) || ~isfinite(val) || val < 0
    error('rsa_meeg:sim:ClusterSpec', 'Cluster node parameter must be a finite nonnegative scalar.');
end
if is_leaf_node(node)
    n_items = node{2};
    if ~isscalar(n_items) || ~isnumeric(n_items) || ~isreal(n_items) || ~isfinite(n_items) || ...
            n_items < 1 || n_items ~= floor(n_items)
        error('rsa_meeg:sim:ClusterSpec', 'Leaf condition count must be a positive integer.');
    end
    if double(val) <= 0
        error('rsa_meeg:sim:ClusterSpec', 'Leaf parameter must be strictly positive.');
    end
else
    for m = 2:numel(node)
        if ~iscell(node{m})
            error('rsa_meeg:sim:ClusterSpec', 'Internal node children must be cell arrays.');
        end
        validate_tree_node(node{m});
    end
end
end

function tf = is_leaf_node(node)
tf = (numel(node) == 2) && isnumeric(node{2}) && isscalar(node{2});
end

function vals = collect_leaf_values(node)
if is_leaf_node(node)
    vals = double(node{1});
else
    vals = [];
    for m = 2:numel(node)
        vals = [vals, collect_leaf_values(node{m})]; %#ok<AGROW>
    end
end
end

function C = build_spread_covariance(node, inherited_var)
% In hierarchical spread/variance-component parameterization:
%   - A leaf {v_leaf, n} has exemplar variance v_leaf * I_n plus shared
%     cluster variance inherited_var * ones(n, n).
%   - An internal node {v_node, child_1, ..., child_M} passes shared
%     variance v_node to each of its child clusters (plus any inherited_var
%     shared across the whole node).
val = double(node{1});
if is_leaf_node(node)
    n = double(node{2});
    C = inherited_var * ones(n, n) + val * eye(n);
else
    blocks = cell(1, numel(node) - 1);
    for m = 2:numel(node)
        child = node{m};
        if is_leaf_node(child)
            blocks{m - 1} = build_spread_covariance(child, val);
        else
            blocks{m - 1} = build_spread_covariance(child, 0.0) + val * ones(count_leaves(child));
        end
    end
    C = blkdiag(blocks{:});
    if inherited_var > 0
        C = C + inherited_var * ones(size(C));
    end
end
end

function n = count_leaves(node)
if is_leaf_node(node)
    n = double(node{2});
else
    n = 0;
    for m = 2:numel(node)
        n = n + count_leaves(node{m});
    end
end
end

function C = build_correlation_covariance(node)
val = double(node{1});
if is_leaf_node(node)
    n = double(node{2});
    if val >= 1
        error('rsa_meeg:sim:ClusterSpec', 'In correlation mode, correlation values must be < 1.');
    end
    C = val * ones(n, n) + (1 - val) * eye(n);
else
    n_children = numel(node) - 1;
    blocks = cell(1, n_children);
    sizes = zeros(1, n_children);
    for m = 1:n_children
        blocks{m} = build_correlation_covariance(node{m + 1});
        sizes(m) = size(blocks{m}, 1);
    end
    total_n = sum(sizes);
    C = val * ones(total_n, total_n);
    idx = 0;
    for m = 1:n_children
        rng_m = (idx + 1):(idx + sizes(m));
        C(rng_m, rng_m) = blocks{m};
        idx = idx + sizes(m);
    end
end
end
