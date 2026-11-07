function [noise_data, info] = cortical_perlin_noise(positions_mm, n_observations, n_times, varargin)
%CORTICAL_PERLIN_NOISE 30-scale cortical geodesic/Euclidean Perlin spatial texture and hybrid sensor noise simulator.
%   [noise_data, info] = rsa_meeg.sim.cortical_perlin_noise(positions_mm, n_observations, n_times, ...)
%
%   Generates multi-scale continuous spatial texture noise across cortical
%   vertices or sensors (default 30 spatial scales r_1..r_30 with 1/f spectral
%   weighting and Gaussian smoothing kernels), and optionally mixes it with
%   correlated prestimulus sensor noise via the Cholesky factor L_m * z of an
%   empirical sensor noise covariance matrix C_m.
%
%   Inputs:
%     positions_mm   - [n_vertices x 3] finite real vertex/sensor coordinates in mm.
%     n_observations - Positive integer number of observations (trials/epochs).
%     n_times        - Positive integer number of time samples per observation (default 1).
%
%   Name-Value Options:
%     'ScalesMm'                   - Vector of positive spatial smoothing scales in mm
%                                    (default linspace(2, 60, 30), i.e., 30 scales).
%     'SpectralSlope'              - Real scalar 1/f^alpha spatial weighting exponent
%                                    across scales, w_k ~ r_k^alpha (default 1.0).
%     'DistanceType'               - 'euclidean' (default) or 'geodesic'. Geodesic
%                                    mode computes shortest-path distances along the
%                                    cortical triangle mesh supplied in 'Triangles'.
%     'Triangles'                  - Optional [n_faces x 3] 1-based triangular mesh
%                                    faces (required when DistanceType is 'geodesic').
%     'DistanceMatrix'             - Optional precomputed [n_vertices x n_vertices]
%                                    nonnegative symmetric distance matrix in mm.
%     'Hemisphere'                 - Optional [n_vertices x 1] text labels ('lh'/'rh')
%                                    enforcing zero spatial leakage across hemispheres.
%     'TemporalSmoothSigmaSamples' - Nonnegative scalar Gaussian temporal smoothing
%                                    sigma in samples (default 0).
%     'Leadfield'                  - Optional [n_sensors x n_vertices] forward matrix G_m.
%     'InverseOperator'            - Optional [n_vertices x n_sensors] inverse matrix K_m.
%     'SensorNoiseCov'             - Optional [n_sensors x n_sensors] prestimulus noise
%                                    covariance C_m (or [n_vertices x n_vertices] if no
%                                    Leadfield/InverseOperator is supplied).
%     'SensorNoiseChol'            - Optional lower-triangular Cholesky factor L_m
%                                    such that L_m * L_m' == C_m.
%     'HybridMix'                  - Scalar in [0, 1] specifying the variance fraction
%                                    of cortical Perlin noise vs. sensor noise L_m * z
%                                    when SensorNoiseCov/Chol is provided (default 0.5).
%     'Amplitude'                  - Positive scalar overall RMS multiplier (default 1.0).
%     'Seed'                       - Optional nonnegative integer RNG seed.
%
%   Outputs:
%     noise_data - [n_observations x n_features_out x n_times] double noise array.
%     info       - Metadata struct containing scales_mm, scale_weights,
%                  distance_type, spectral_slope, hybrid_mix, spatial_covariance,
%                  and sensor_chol.

if nargin < 3 || isempty(n_times)
    n_times = 1;
end
if ~isnumeric(positions_mm) || ~isreal(positions_mm) || size(positions_mm, 2) ~= 3 || ...
        isempty(positions_mm) || any(~isfinite(positions_mm(:)))
    error('rsa_meeg:sim:Positions', 'positions_mm must be finite N x 3 coordinates in mm.');
end
if ~isscalar(n_observations) || ~isnumeric(n_observations) || ~isreal(n_observations) || ...
        ~isfinite(n_observations) || n_observations < 1 || n_observations ~= floor(n_observations)
    error('rsa_meeg:sim:Observations', 'n_observations must be a positive integer scalar.');
end
if ~isscalar(n_times) || ~isnumeric(n_times) || ~isreal(n_times) || ...
        ~isfinite(n_times) || n_times < 1 || n_times ~= floor(n_times)
    error('rsa_meeg:sim:Times', 'n_times must be a positive integer scalar.');
end

n_vertices = size(positions_mm, 1);
n_observations = double(n_observations);
n_times = double(n_times);
positions_mm = double(positions_mm);

p = inputParser;
p.FunctionName = 'rsa_meeg.sim.cortical_perlin_noise';
addParameter(p, 'ScalesMm', linspace(2, 60, 30));
addParameter(p, 'SpectralSlope', 1.0);
addParameter(p, 'DistanceType', 'euclidean');
addParameter(p, 'Triangles', []);
addParameter(p, 'DistanceMatrix', []);
addParameter(p, 'Hemisphere', []);
addParameter(p, 'TemporalSmoothSigmaSamples', 0);
addParameter(p, 'Leadfield', []);
addParameter(p, 'InverseOperator', []);
addParameter(p, 'SensorNoiseCov', []);
addParameter(p, 'SensorNoiseChol', []);
addParameter(p, 'HybridMix', 0.5);
addParameter(p, 'Amplitude', 1.0);
addParameter(p, 'Seed', []);
parse(p, varargin{:});
o = p.Results;

scales_mm = double(o.ScalesMm(:).');
if isempty(scales_mm) || ~isnumeric(scales_mm) || ~isreal(scales_mm) || any(~isfinite(scales_mm)) || any(scales_mm <= 0)
    error('rsa_meeg:sim:ScalesMm', 'ScalesMm must be a nonempty vector of positive finite scales in mm.');
end
if ~isscalar(o.SpectralSlope) || ~isnumeric(o.SpectralSlope) || ~isreal(o.SpectralSlope) || ~isfinite(o.SpectralSlope)
    error('rsa_meeg:sim:SpectralSlope', 'SpectralSlope must be a finite real scalar.');
end
dist_type = lower(string(o.DistanceType));
if ~isscalar(dist_type) || ~any(dist_type == ["euclidean", "geodesic"])
    error('rsa_meeg:sim:DistanceType', 'DistanceType must be euclidean or geodesic.');
end
if ~isscalar(o.TemporalSmoothSigmaSamples) || ~isnumeric(o.TemporalSmoothSigmaSamples) || ...
        ~isreal(o.TemporalSmoothSigmaSamples) || ~isfinite(o.TemporalSmoothSigmaSamples) || o.TemporalSmoothSigmaSamples < 0
    error('rsa_meeg:sim:TemporalSmooth', 'TemporalSmoothSigmaSamples must be a nonnegative finite scalar.');
end
if ~isscalar(o.HybridMix) || ~isnumeric(o.HybridMix) || ~isreal(o.HybridMix) || ...
        ~isfinite(o.HybridMix) || o.HybridMix < 0 || o.HybridMix > 1
    error('rsa_meeg:sim:HybridMix', 'HybridMix must be a scalar in [0, 1].');
end
if ~isscalar(o.Amplitude) || ~isnumeric(o.Amplitude) || ~isreal(o.Amplitude) || ...
        ~isfinite(o.Amplitude) || o.Amplitude <= 0
    error('rsa_meeg:sim:Amplitude', 'Amplitude must be a positive finite scalar.');
end
if ~isempty(o.Seed)
    if ~isscalar(o.Seed) || ~isnumeric(o.Seed) || ~isreal(o.Seed) || ~isfinite(o.Seed) || o.Seed < 0 || o.Seed ~= floor(o.Seed)
        error('rsa_meeg:sim:Seed', 'Seed must be a nonnegative integer scalar.');
    end
    prior_rng = rng;
    cleanup_rng = onCleanup(@() rng(prior_rng));
    rng(double(o.Seed), 'twister');
end

% 1. Compute pairwise cortical distance matrix D (in mm)
if ~isempty(o.DistanceMatrix)
    D = double(o.DistanceMatrix);
    if ~isequal(size(D), [n_vertices, n_vertices]) || any(isnan(D(:))) || any(D(:) < 0)
        error('rsa_meeg:sim:DistanceMatrix', 'DistanceMatrix must be a nonnegative N x N matrix.');
    end
elseif dist_type == "euclidean"
    diff_x = bsxfun(@minus, positions_mm(:, 1), positions_mm(:, 1).');
    diff_y = bsxfun(@minus, positions_mm(:, 2), positions_mm(:, 2).');
    diff_z = bsxfun(@minus, positions_mm(:, 3), positions_mm(:, 3).');
    D = sqrt(diff_x.^2 + diff_y.^2 + diff_z.^2);
else
    if isempty(o.Triangles)
        error('rsa_meeg:sim:TrianglesRequired', 'Triangles must be supplied when DistanceType is geodesic.');
    end
    A_adj = rsa_meeg.geometry.mesh_adjacency(o.Triangles, n_vertices);
    [u_idx, v_idx] = find(triu(A_adj, 1));
    edge_len = sqrt(sum((positions_mm(u_idx, :) - positions_mm(v_idx, :)).^2, 2));
    W_graph = sparse([u_idx; v_idx], [v_idx; u_idx], [edge_len; edge_len], n_vertices, n_vertices);
    G_mesh = graph(W_graph);
    D = distances(G_mesh);
end

if ~isempty(o.Hemisphere)
    if ~(isstring(o.Hemisphere) || iscellstr(o.Hemisphere)) || numel(o.Hemisphere) ~= n_vertices
        error('rsa_meeg:sim:Hemisphere', 'Hemisphere must provide one label per vertex.');
    end
    hemi = string(o.Hemisphere(:));
    cross_hemi = bsxfun(@ne, hemi, hemi.');
    D(cross_hemi) = inf;
end

% 2. Multi-scale Perlin kernel synthesis across all scales r_1 .. r_S
n_scales = numel(scales_mm);
raw_weights = scales_mm .^ double(o.SpectralSlope);
scale_weights = raw_weights / sqrt(sum(raw_weights.^2));

n_total_cols = n_observations * n_times;
perlin_flat = zeros(n_vertices, n_total_cols);
spatial_cov = zeros(n_vertices, n_vertices);

for k = 1:n_scales
    rk = scales_mm(k);
    K_k = exp(-0.5 * (D ./ rk).^2);
    K_k(isnan(K_k)) = 0;
    row_norms = sqrt(sum(K_k.^2, 2));
    row_norms(row_norms == 0) = 1;
    K_k = bsxfun(@rdivide, K_k, row_norms);
    Zk = randn(n_vertices, n_total_cols);
    perlin_flat = perlin_flat + scale_weights(k) * (K_k * Zk);
    if nargout > 1
        spatial_cov = spatial_cov + (scale_weights(k)^2) * (K_k * K_k.');
    end
end

% Optional temporal Gaussian smoothing across samples within each observation
if o.TemporalSmoothSigmaSamples > 0 && n_times > 1
    t_grid = 1:n_times;
    t_diff = bsxfun(@minus, t_grid(:), t_grid(:).');
    Kt = exp(-0.5 * (t_diff ./ double(o.TemporalSmoothSigmaSamples)).^2);
    Kt = bsxfun(@rdivide, Kt, sqrt(sum(Kt.^2, 2)));
    perlin_3d = reshape(perlin_flat, [n_vertices, n_observations, n_times]);
    for obs = 1:n_observations
        slice = reshape(perlin_3d(:, obs, :), [n_vertices, n_times]);
        perlin_3d(:, obs, :) = slice * Kt.';
    end
    perlin_flat = reshape(perlin_3d, [n_vertices, n_total_cols]);
end

% 3. Optional forward/inverse projection and hybrid mixing with prestimulus sensor noise L_m * z
has_G = ~isempty(o.Leadfield);
has_K = ~isempty(o.InverseOperator);
has_cov = ~isempty(o.SensorNoiseCov) || ~isempty(o.SensorNoiseChol);

if has_G
    G_m = double(o.Leadfield);
    if ~isreal(G_m) || size(G_m, 2) ~= n_vertices || any(~isfinite(G_m(:)))
        error('rsa_meeg:sim:Leadfield', 'Leadfield must be finite real [n_sensors x n_vertices].');
    end
    n_sensors = size(G_m, 1);
else
    n_sensors = n_vertices;
end

if has_K
    K_m = double(o.InverseOperator);
    if ~isreal(K_m) || size(K_m, 1) ~= n_vertices || size(K_m, 2) ~= n_sensors || any(~isfinite(K_m(:)))
        error('rsa_meeg:sim:InverseOperator', 'InverseOperator must be finite real [n_vertices x n_sensors].');
    end
end

L_m = [];
if has_cov
    if ~isempty(o.SensorNoiseChol)
        L_m = double(o.SensorNoiseChol);
        if ~isreal(L_m) || size(L_m, 1) ~= n_sensors || any(~isfinite(L_m(:)))
            error('rsa_meeg:sim:SensorNoiseChol', 'SensorNoiseChol must be finite real with n_sensors rows.');
        end
    else
        C_m = double(o.SensorNoiseCov);
        if ~isreal(C_m) || ~isequal(size(C_m), [n_sensors, n_sensors]) || any(~isfinite(C_m(:)))
            error('rsa_meeg:sim:SensorNoiseCov', 'SensorNoiseCov must be finite real [n_sensors x n_sensors].');
        end
        if norm(C_m - C_m.', 'fro') > 1e-10 * max(1, norm(C_m, 'fro'))
            error('rsa_meeg:sim:SensorNoiseCov', 'SensorNoiseCov must be symmetric.');
        end
        C_m = 0.5 * (C_m + C_m.');
        [R_chol, flag] = chol(C_m, 'lower');
        if flag == 0
            L_m = R_chol;
        else
            % Support rank-deficient projected covariances (e.g., average-referenced EEG)
            [V_c, D_c] = eig(C_m);
            ev_c = real(diag(D_c));
            if min(ev_c) < -1e-8 * max(1, max(ev_c))
                error('rsa_meeg:sim:SensorNoiseCov', 'SensorNoiseCov must be positive semidefinite.');
            end
            ev_c = max(ev_c, 0);
            L_m = real(V_c) * diag(sqrt(ev_c));
        end
    end
end

if has_G && has_K
    cortical_proj = (K_m * G_m) * perlin_flat;
elseif has_G && ~has_K
    cortical_proj = G_m * perlin_flat;
else
    cortical_proj = perlin_flat;
end

rms_cort = sqrt(mean(cortical_proj(:).^2));
if rms_cort > 0
    cortical_proj = cortical_proj / rms_cort;
end

if has_cov
    Z_sens = randn(size(L_m, 2), n_total_cols);
    sens_noise = L_m * Z_sens;
    if has_K
        sens_proj = K_m * sens_noise;
    else
        sens_proj = sens_noise;
    end
    rms_sens = sqrt(mean(sens_proj(:).^2));
    if rms_sens > 0
        sens_proj = sens_proj / rms_sens;
    end
    gamma = double(o.HybridMix);
    combined_flat = sqrt(gamma) * cortical_proj + sqrt(1 - gamma) * sens_proj;
    effective_mix = gamma;
else
    combined_flat = cortical_proj;
    effective_mix = 1.0;
end

combined_flat = double(o.Amplitude) * combined_flat;
n_feat_out = size(combined_flat, 1);
noise_3d = reshape(combined_flat, [n_feat_out, n_observations, n_times]);
noise_data = permute(noise_3d, [2, 1, 3]);

info = struct();
info.scales_mm = scales_mm;
info.scale_weights = scale_weights;
info.distance_type = char(dist_type);
info.spectral_slope = double(o.SpectralSlope);
info.hybrid_mix = effective_mix;
info.spatial_covariance = 0.5 * (spatial_cov + spatial_cov.');
info.sensor_chol = L_m;
end
