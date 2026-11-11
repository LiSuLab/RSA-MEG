function tests = test_v2_new_features
%TEST_V2_NEW_FEATURES Unit and integration tests for RSA-MEEG V2 upgrades:
%   1. rsa_meeg.sim.generate_beta_patterns
%   2. rsa_meeg.sim.cortical_perlin_noise
%   3. rsa_meeg.rdm.shrinkage_precision
%   4. rsa_meeg.mapping.partial_compare
%   5. demo_eeg_meg_spatial_rescue
tests = functiontests(localfunctions);
end

function test_generate_beta_patterns_hierarchy_and_unbiasedness(testCase)
% Verify canonical 4x5 hierarchical cluster tree {0.6, {0.3, 5}, ...}
spec = {0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}};
n_feat = 60;
[B, cov_true, rdm_true] = rsa_meeg.sim.generate_beta_patterns( ...
    spec, n_feat, 'ExactCovariance', true, 'Seed', 42);

verifyEqual(testCase, size(B), [20, n_feat]);
verifyEqual(testCase, size(cov_true), [20, 20]);
verifyEqual(testCase, size(rdm_true.values), [190, 1]);
verifyEqual(testCase, rdm_true.condition_ids, 1:20);
verifyEqual(testCase, size(rdm_true.square), [20, 20]);

% ExactCovariance=true must match sample covariance (B*B')/n_feat to machine precision
cov_emp = (B * B.') / n_feat;
verifyLessThan(testCase, norm(cov_emp - cov_true, 'fro'), 1e-11);
verifyLessThan(testCase, max(abs(diag(cov_true) - 1)), 1e-12);
verifyGreaterThan(testCase, min(eig(cov_true)), 0);

% Within-cluster correlation must be 0.6 / (0.6 + 0.3) = 2/3, between-cluster = 0
verifyLessThan(testCase, abs(cov_true(1, 2) - 2/3), 1e-12);
verifyLessThan(testCase, abs(cov_true(1, 6) - 0.0), 1e-12);

% Within-cluster RDM distance must be strictly smaller than between-cluster distance
verifyLessThan(testCase, rdm_true.square(1, 2), rdm_true.square(1, 6));

% Monte Carlo unbiasedness when ExactCovariance=false
[~, cov_mc_true, rdm_mc_true] = rsa_meeg.sim.generate_beta_patterns( ...
    spec, 4000, 'ExactCovariance', false, 'Seed', 99);
[B_mc, ~, ~] = rsa_meeg.sim.generate_beta_patterns( ...
    spec, 4000, 'ExactCovariance', false, 'Seed', 99);
cov_mc_emp = (B_mc * B_mc.') / 4000;
verifyLessThan(testCase, max(abs(cov_mc_emp(:) - cov_mc_true(:))), 0.06);

% Correlation-mode tree {0.2, {0.7, 4}, {0.7, 4}} and 3-level nested tree
spec_corr = {0.2, {0.7, 4}, {0.7, 4}};
[B_c, cov_c, ~] = rsa_meeg.sim.generate_beta_patterns(spec_corr, 20, 'ExactCovariance', true, 'Seed', 7);
verifyEqual(testCase, size(B_c), [8, 20]);
verifyLessThan(testCase, abs(cov_c(1, 2) - 0.7), 1e-12);
verifyLessThan(testCase, abs(cov_c(1, 5) - 0.2), 1e-12);

spec_3lvl = {0.8, {0.4, {0.2, 3}, {0.2, 3}}, {0.4, {0.2, 3}, {0.2, 3}}};
[B_3, cov_3, ~] = rsa_meeg.sim.generate_beta_patterns(spec_3lvl, 30, 'ExactCovariance', true, 'Seed', 11);
verifyEqual(testCase, size(B_3), [12, 30]);
verifyGreaterThan(testCase, cov_3(1, 2), cov_3(1, 4));
verifyGreaterThan(testCase, cov_3(1, 4), cov_3(1, 7));

% Boundary error checks
verifyError(testCase, @() rsa_meeg.sim.generate_beta_patterns(spec, 5, 'ExactCovariance', true), ...
    'rsa_meeg:sim:ExactCovarianceDim');
verifyError(testCase, @() rsa_meeg.sim.generate_beta_patterns({-0.5, 4}, 10), ...
    'rsa_meeg:sim:ClusterSpec');
end

function test_cortical_perlin_noise_multiscale_and_hybrid(testCase)
% Construct a 2-hemisphere triangular surface mesh (12 vertices, 6 per hemisphere)
pos_lh = [0 0 0; 10 0 0; 5 8 0; 15 8 0; 10 16 0; 20 16 0];
pos_rh = pos_lh + [100 0 0];
positions_mm = [pos_lh; pos_rh];
triangles = [1 2 3; 2 4 3; 3 4 5; 4 6 5; ...
             7 8 9; 8 10 9; 9 10 11; 10 12 11];
hemi = [repmat({'lh'}, 6, 1); repmat({'rh'}, 6, 1)];

% 30-scale Euclidean Perlin noise with hemisphere isolation
[noise_euc, info_euc] = rsa_meeg.sim.cortical_perlin_noise( ...
    positions_mm, 25, 10, ...
    'ScalesMm', linspace(2, 60, 30), ...
    'SpectralSlope', 1.0, ...
    'Hemisphere', hemi, ...
    'TemporalSmoothSigmaSamples', 1.5, ...
    'Seed', 123);

verifyEqual(testCase, size(noise_euc), [25, 12, 10]);
verifyEqual(testCase, numel(info_euc.scales_mm), 30);
% Cross-hemisphere theoretical spatial covariance must be identically zero
verifyLessThan(testCase, max(max(abs(info_euc.spatial_covariance(1:6, 7:12)))), 1e-14);
% Nearby vertices within hemisphere have positive spatial covariance
verifyGreaterThan(testCase, info_euc.spatial_covariance(1, 2), info_euc.spatial_covariance(1, 6));

% Geodesic Perlin noise on cortical mesh
[noise_geo, info_geo] = rsa_meeg.sim.cortical_perlin_noise( ...
    positions_mm, 10, 4, ...
    'DistanceType', 'geodesic', ...
    'Triangles', triangles, ...
    'Hemisphere', hemi, ...
    'Seed', 456);
verifyEqual(testCase, size(noise_geo), [10, 12, 4]);
verifyEqual(testCase, info_geo.distance_type, 'geodesic');

% Hybrid mixing with rank-deficient projected EEG sensor covariance L_m * z
n_sens = 8;
G_m = randn(n_sens, 12);
K_m = pinv(G_m);
% Average-reference projector makes C_sens rank n_sens - 1
P_avg = eye(n_sens) - ones(n_sens) / n_sens;
A_rand = randn(n_sens);
C_sens = P_avg * (A_rand * A_rand.' + 0.5 * eye(n_sens)) * P_avg;

[noise_hyb, info_hyb] = rsa_meeg.sim.cortical_perlin_noise( ...
    positions_mm, 15, 6, ...
    'Leadfield', G_m, ...
    'InverseOperator', K_m, ...
    'SensorNoiseCov', C_sens, ...
    'HybridMix', 0.4, ...
    'Amplitude', 2.0, ...
    'Seed', 789);
verifyEqual(testCase, size(noise_hyb), [15, 12, 6]);
verifyEqual(testCase, info_hyb.hybrid_mix, 0.4, 'AbsTol', 1e-12);
verifyFalse(testCase, isempty(info_hyb.sensor_chol));

% Boundary error checks
verifyError(testCase, @() rsa_meeg.sim.cortical_perlin_noise(positions_mm, 5, 2, 'DistanceType', 'geodesic'), ...
    'rsa_meeg:sim:TrianglesRequired');
verifyError(testCase, @() rsa_meeg.sim.cortical_perlin_noise(positions_mm, 5, 2, 'HybridMix', 1.5), ...
    'rsa_meeg:sim:HybridMix');
end

function test_shrinkage_precision_methods_and_searchlight_integration(testCase)
% 1. High-dimensional regime (D > N): verify strict positive-definiteness across all methods
rng(555, 'twister');
N = 12;
D = 30;
X_res = randn(N, D);
for m = ["ledoit-wolf", "oas", "diagonal"]
    [P_m, info_m] = rsa_meeg.rdm.shrinkage_precision(X_res, 'Method', m, 'SubtractConditionMeans', false);
    verifyEqual(testCase, size(P_m), [D, D]);
    verifyLessThan(testCase, norm(P_m - P_m.', 'fro'), 1e-12);
    verifyGreaterThan(testCase, min(eig(P_m)), 0);
    verifyGreaterThanOrEqual(testCase, info_m.shrinkage, 0);
    verifyLessThanOrEqual(testCase, info_m.shrinkage, 1);
end

[P_fix, info_fix] = rsa_meeg.rdm.shrinkage_precision(X_res, 'Shrinkage', 0.25, 'SubtractConditionMeans', false);
verifyEqual(testCase, info_fix.shrinkage, 0.25, 'AbsTol', 1e-12);
verifyGreaterThan(testCase, min(eig(P_fix)), 0);

% 2. Integration with rsa_meeg.rdm.searchlight in spatiotemporal and spatial modes
times = -0.05:0.01:0.05;
nf = 8;
conds = repmat((1:4).', 4, 1);
folds = repelem([1; 2], 8);
pos = [(0:nf-1).' * 5, zeros(nf, 2)];
hood = rsa_meeg.geometry.neighborhoods(pos, 11);

% Under pure Gaussian noise, crossnobis RDM with shrinkage_precision must be unbiased (mean ~ 0)
null_means = zeros(20, 1);
for rep = 1:20
    data_null = randn(numel(conds), nf, numel(times));
    ds_null = rsa_meeg.dataset(data_null, times, conds, 'FoldIds', folds, ...
        'FeatureIds', 1:nf, 'FeatureTypes', repmat({'eeg'}, 1, nf), 'Positions', pos);
    P_cell = rsa_meeg.rdm.shrinkage_precision(ds_null, hood, ...
        'Method', 'ledoit-wolf', 'Mode', 'spatiotemporal', 'WindowSamples', 3);
    R_null = rsa_meeg.rdm.searchlight(ds_null, hood, ...
        'WindowSamples', 3, 'StepSamples', 3, 'Mode', 'spatiotemporal', ...
        'Metric', 'crossnobis', 'Precision', P_cell);
    null_means(rep) = mean(R_null.values(:));
end
verifyLessThan(testCase, abs(mean(null_means)), 0.05);

% Verify BaselineWindow option
P_base = rsa_meeg.rdm.shrinkage_precision(ds_null, hood, ...
    'Method', 'oas', 'Mode', 'spatial', 'BaselineWindow', [-0.05, 0.0]);
verifyEqual(testCase, numel(P_base), nf);
end

function test_partial_compare_orthogonality_and_invariance(testCase)
% Construct a 6-condition RDM scenario (15 pairs)
rng(777, 'twister');
nc = 6;
pairs = nchoosek(1:nc, 2);
np = size(pairs, 1);

% True independent latent factors
s_vec = [1; 2; 4; 7; 11; 16];
c_vec = [10; -5; 8; -3; 6; -9];
sig_rdm = abs(s_vec(pairs(:, 1)) - s_vec(pairs(:, 2)));
ctrl_rdm = abs(c_vec(pairs(:, 1)) - c_vec(pairs(:, 2)));

% Target model is contaminated by control model: target = sig + 2.5 * ctrl
target_vals = sig_rdm + 2.5 * ctrl_rdm;

% Center 1 neural RDM has ONLY the control confound (no true signal)
% Center 2 neural RDM has pure signal + control confound
R = struct();
R.condition_ids = 1:nc;
R.pairs = pairs;
R.center_ids = [10, 20];
R.windows = struct('start_s', 0, 'center_s', 0, 'end_s', 0, 'indices', {{1}});
R.values = zeros(np, 2, 1);
R.values(:, 1, 1) = 3.0 * ctrl_rdm;
R.values(:, 2, 1) = 1.5 * sig_rdm + 2.0 * ctrl_rdm;

% Uncontrolled Pearson correlation at Center 1 is spuriously positive,
% whereas partial_compare controlling for ctrl_rdm yields exactly 0 at Center 1
% and 1.0 at Center 2!
raw_fit = rsa_meeg.mapping.compare(R, target_vals, 'Correlation', 'pearson');
verifyGreaterThan(testCase, raw_fit.values(1, 1, 1), 0.5);

part_pearson = rsa_meeg.mapping.partial_compare(R, target_vals, ctrl_rdm, 'Correlation', 'pearson');
verifyEqual(testCase, part_pearson.values(1, 1, 1), 0.0, 'AbsTol', 1e-12);
verifyEqual(testCase, part_pearson.values(2, 1, 1), 1.0, 'AbsTol', 1e-12);

% Verify Spearman partial correlation and condition-label permutation invariance
perm = [4, 2, 6, 1, 5, 3];
perm_pairs = nchoosek(1:nc, 2);
target_perm_vals = zeros(np, 1);
ctrl_perm_vals = zeros(np, 1);
for k = 1:np
    a = sort([perm(perm_pairs(k, 1)), perm(perm_pairs(k, 2))]);
    idx = find(pairs(:, 1) == a(1) & pairs(:, 2) == a(2), 1);
    target_perm_vals(k) = target_vals(idx);
    ctrl_perm_vals(k) = ctrl_rdm(idx);
end
target_struct = struct('condition_ids', perm, 'values', target_perm_vals);
ctrl_struct = struct('condition_ids', perm, 'values', ctrl_perm_vals);

part_sp_direct = rsa_meeg.mapping.partial_compare(R, target_vals, ctrl_rdm, 'Correlation', 'spearman');
part_sp_struct = rsa_meeg.mapping.partial_compare(R, target_struct, {ctrl_struct}, 'Correlation', 'spearman');
verifyEqual(testCase, part_sp_direct.values, part_sp_struct.values, 'AbsTol', 1e-12);
verifyEqual(testCase, part_sp_direct.values(1, 1, 1), 0.0, 'AbsTol', 1e-12);

% Rank-deficient control model must raise an error
verifyError(testCase, @() rsa_meeg.mapping.partial_compare(R, target_vals, [ctrl_rdm, 2 * ctrl_rdm]), ...
    'rsa_meeg:mapping:PartialRank');
end

function test_demo_eeg_meg_spatial_rescue_execution(testCase)
out = demo_eeg_meg_spatial_rescue(false);
verifyGreaterThan(testCase, out.isf_mean_r, 0.5);
verifyLessThan(testCase, abs(out.asf_mean_r), 0.2);
verifyGreaterThan(testCase, out.isf_mean_r, out.asf_mean_r);
verifyGreaterThan(testCase, out.focal_specificity_rescued, out.focal_specificity_unwhitened);
end

function test_rsa_namespace_alias_and_64condition_v2(testCase)
% Verify +rsa (+sim, +rdm, +mapping, +stats) forwarding aliases and 64-condition (4x16) V2 hierarchy
spec64 = {0.6, {0.3, 16}, {0.3, 16}, {0.3, 16}, {0.3, 16}};
[B64, cov64, rdm64] = rsa.sim.generate_beta_patterns(spec64, 80, 'ExactCovariance', true, 'Seed', 2026);
verifyEqual(testCase, size(B64), [64, 80]);
verifyEqual(testCase, size(cov64), [64, 64]);
verifyEqual(testCase, size(rdm64.values), [2016, 1]);

pos = [(0:5).' * 5, zeros(6, 2)];
[noise, info] = rsa.sim.cortical_perlin_noise(pos, 16, 2, 'Seed', 11);
verifyEqual(testCase, size(noise), [16, 6, 2]);
verifyEqual(testCase, numel(info.scales_mm), 30);

[P, pinfo] = rsa.rdm.shrinkage_precision(noise(:, :, 1), 'Method', 'ledoit-wolf');
verifyEqual(testCase, size(P), [6, 6]);
verifyGreaterThanOrEqual(testCase, pinfo.shrinkage, 0);

x_stat = [2 2; 3 3; 4 4; 5 5];
g = rsa.stats.group(x_stat, 'Exact', true);
verifyEqual(testCase, g.n_permutations, 16);
end
