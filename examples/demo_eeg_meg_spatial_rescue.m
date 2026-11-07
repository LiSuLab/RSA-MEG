function out = demo_eeg_meg_spatial_rescue(make_figures)
%DEMO_EEG_MEG_SPATIAL_RESCUE Self-contained V2 demonstration of:
%   (1) Hierarchical 20-condition multivariate pattern generation
%       (rsa_meeg.sim.generate_beta_patterns);
%   (2) 30-scale cortical Perlin spatial texture and hybrid sensor noise
%       (rsa_meeg.sim.cortical_perlin_noise);
%   (3) Multivariate ISF (RSA) vs. univariate ASF (mean activation RDM); and
%   (4) EEG spatial super-resolution rescue via spatiotemporal searchlight +
%       Ledoit-Wolf shrinkage precision (rsa_meeg.rdm.shrinkage_precision) and
%       partial RDM correlation (rsa_meeg.mapping.partial_compare).
%
%   Usage:
%     out = demo_eeg_meg_spatial_rescue(false);

if nargin < 1
    make_figures = true;
end
prior_rng = rng;
cleanup_rng = onCleanup(@() rng(prior_rng));
rng(20261104, 'twister');

% 1. Cortical geometry: 16 vertices along a 1-D auditory bank (5 mm spacing)
%    Focal coding patch (Heschl's gyrus): vertices 6..10
%    Flanking non-coding bystander regions (lateral STG): vertices 1..5, 11..16
n_vertices = 16;
positions_mm = [(0:n_vertices-1).' * 5, zeros(n_vertices, 2)];
hemi = repmat({'lh'}, n_vertices, 1);
target_vertices = 6:10;
flanking_vertices = setdiff(1:n_vertices, target_vertices);
hood = rsa_meeg.geometry.neighborhoods(positions_mm, 12, 'Hemisphere', hemi);

% 2. Hierarchical 20-condition ground-truth patterns (4 clusters x 5 items)
cluster_spec = {0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}};
n_target = numel(target_vertices);
[B_target, cov_true, rdm_true] = rsa_meeg.sim.generate_beta_patterns( ...
    cluster_spec, n_target, 'ExactCovariance', false, 'Seed', 101);

% Center each condition pattern across target vertices so univariate mean
% activation is zero (purely multivariate spatial reconfiguration code)
B_target_centered = bsxfun(@minus, B_target, mean(B_target, 2));
B_full = zeros(size(B_target, 1), n_vertices);
B_full(:, target_vertices) = 2.2 * B_target_centered;

% Construct a nuisance covariate RDM (e.g., low-level stimulus energy) orthogonalized or distinct
n_cond = size(B_target, 1);
nuisance_vec = sin((1:n_cond).' * 0.7);
pairs = rdm_true.pairs;
nuisance_model = struct();
nuisance_model.condition_ids = 1:n_cond;
nuisance_model.values = abs(nuisance_vec(pairs(:, 1)) - nuisance_vec(pairs(:, 2)));

% 3. Forward/inverse spatial resolution operators:
%    - MEG Gradiometer benchmark: sharp focal spatial resolution (sigma = 5 mm)
%    - EEG Volume Conduction: broad spatial blurring (sigma = 18 mm) + strong
%      common-mode crosstalk across neighboring vertices
dist_mat = abs(bsxfun(@minus, positions_mm(:, 1), positions_mm(:, 1).'));
R_meg = exp(-0.5 * (dist_mat / 5.0).^2);
R_meg = bsxfun(@rdivide, R_meg, sum(R_meg, 2));

% EEG resolution operator mixes focal gradient with broad common-mode smear
R_eeg_broad = exp(-0.5 * (dist_mat / 22.0).^2);
R_eeg_broad = bsxfun(@rdivide, R_eeg_broad, sum(R_eeg_broad, 2));
R_eeg = 0.25 * eye(n_vertices) + 0.75 * R_eeg_broad;

% Common-mode volume-conduction noise covariance in EEG space
C_eeg_vc = 0.15 * eye(n_vertices) + 1.85 * exp(-0.5 * (dist_mat / 25.0).^2);

% 4. Simulate trials across 2 independent folds x 2 repeats per fold (80 epochs)
sample_times_s = -0.05:0.01:0.15;
n_times = numel(sample_times_s);
evoked_idx = find(sample_times_s >= 0.04 & sample_times_s <= 0.12);
evoked_win_idx = 3; % window centered in the evoked interval
envelope = exp(-0.5 * ((sample_times_s - 0.08) / 0.025).^2);

cond_rep = repmat((1:n_cond).', 4, 1);
fold_rep = repelem([1; 2], 2 * n_cond);
n_obs = numel(cond_rep);

% Generate 30-scale cortical Perlin noise + correlated EEG volume-conduction noise
[noise_eeg, perlin_info] = rsa_meeg.sim.cortical_perlin_noise( ...
    positions_mm, n_obs, n_times, ...
    'ScalesMm', linspace(2, 60, 30), ...
    'SpectralSlope', 1.0, ...
    'SensorNoiseCov', C_eeg_vc, ...
    'HybridMix', 0.25, ...
    'Amplitude', 0.85, ...
    'Seed', 202);

[noise_meg, ~] = rsa_meeg.sim.cortical_perlin_noise( ...
    positions_mm, n_obs, n_times, ...
    'ScalesMm', linspace(2, 60, 30), ...
    'SpectralSlope', 1.0, ...
    'Amplitude', 0.35, ...
    'Seed', 303);

data_eeg = noise_eeg;
data_meg = noise_meg;
for i = 1:n_obs
    c = cond_rep(i);
    sig_meg = (R_meg * B_full(c, :).') * envelope;
    sig_eeg = (R_eeg * B_full(c, :).') * envelope;
    data_meg(i, :, :) = reshape(data_meg(i, :, :), [n_vertices, n_times]) + sig_meg;
    data_eeg(i, :, :) = reshape(data_eeg(i, :, :), [n_vertices, n_times]) + sig_eeg;
end

ds_eeg = rsa_meeg.dataset(data_eeg, sample_times_s, cond_rep, ...
    'FoldIds', fold_rep, 'FeatureIds', 1:n_vertices, ...
    'FeatureTypes', repmat({'source'}, 1, n_vertices), ...
    'Positions', positions_mm, 'Hemisphere', hemi, ...
    'Reference', 'average reference source projection', 'SubjectId', 1);

ds_meg = rsa_meeg.dataset(data_meg, sample_times_s, cond_rep, ...
    'FoldIds', fold_rep, 'FeatureIds', 1:n_vertices, ...
    'FeatureTypes', repmat({'source'}, 1, n_vertices), ...
    'Positions', positions_mm, 'Hemisphere', hemi, ...
    'Reference', 'planar gradiometer source projection', 'SubjectId', 1);

% 5. Comparison A: Multivariate ISF (spatial pattern RSA) vs. Univariate ASF (mean activation RDM)
R_isf = rsa_meeg.rdm.searchlight(ds_meg, {target_vertices}, ...
    'WindowSamples', 5, 'StepSamples', 4, 'Mode', 'spatial', 'Metric', 'sqeuclidean');
R_asf = rsa_meeg.rdm.searchlight(ds_meg, {target_vertices}, ...
    'WindowSamples', 5, 'StepSamples', 4, 'Mode', 'temporal', 'Metric', 'sqeuclidean');

fit_isf = rsa_meeg.mapping.compare(R_isf, rdm_true, 'Correlation', 'spearman');
fit_asf = rsa_meeg.mapping.compare(R_asf, rdm_true, 'Correlation', 'spearman');
isf_mean_r = max(fit_isf.values(:));
asf_mean_r = max(fit_asf.values(:));

% 6. Comparison B: Unwhitened EEG searchlight vs. Shrunk-Precision Crossnobis EEG rescue
R_eeg_unwhitened = rsa_meeg.rdm.searchlight(ds_eeg, hood, ...
    'WindowSamples', 5, 'StepSamples', 4, 'Mode', 'spatial', 'Metric', 'sqeuclidean');

[P_eeg_lw, prec_info] = rsa_meeg.rdm.shrinkage_precision(ds_eeg, hood, ...
    'Method', 'ledoit-wolf', 'Mode', 'spatial', 'WindowSamples', 5);
R_eeg_rescued = rsa_meeg.rdm.searchlight(ds_eeg, hood, ...
    'WindowSamples', 5, 'StepSamples', 4, 'Mode', 'spatial', ...
    'Metric', 'crossnobis', 'Precision', P_eeg_lw);

R_meg_bench = rsa_meeg.rdm.searchlight(ds_meg, hood, ...
    'WindowSamples', 5, 'StepSamples', 4, 'Mode', 'spatial', 'Metric', 'sqeuclidean');

map_eeg_unwhitened = rsa_meeg.mapping.compare(R_eeg_unwhitened, rdm_true, 'Correlation', 'spearman');
map_eeg_rescued = rsa_meeg.mapping.compare(R_eeg_rescued, rdm_true, 'Correlation', 'spearman');
map_eeg_partial = rsa_meeg.mapping.partial_compare(R_eeg_rescued, rdm_true, nuisance_model, 'Correlation', 'spearman');
map_meg_bench = rsa_meeg.mapping.compare(R_meg_bench, rdm_true, 'Correlation', 'spearman');

prof_unwhitened = map_eeg_unwhitened.values(:, evoked_win_idx);
prof_rescued = map_eeg_rescued.values(:, evoked_win_idx);
prof_meg = map_meg_bench.values(:, evoked_win_idx);

% Focal specificity = mean(target vertices) - mean(flanking non-coding vertices)
spec_unwhitened = mean(prof_unwhitened(target_vertices)) - mean(prof_unwhitened(flanking_vertices));
spec_rescued = mean(prof_rescued(target_vertices)) - mean(prof_rescued(flanking_vertices));
spec_meg = mean(prof_meg(target_vertices)) - mean(prof_meg(flanking_vertices));

out = struct();
out.cov_true = cov_true;
out.rdm_true = rdm_true;
out.perlin_info = perlin_info;
out.precision_info = prec_info;
out.isf_mean_r = isf_mean_r;
out.asf_mean_r = asf_mean_r;
out.eeg_unwhitened_profile = prof_unwhitened;
out.eeg_rescued_profile = prof_rescued;
out.meg_benchmark_profile = prof_meg;
out.partial_compare_map = map_eeg_partial;
out.focal_specificity_unwhitened = spec_unwhitened;
out.focal_specificity_rescued = spec_rescued;
out.focal_specificity_meg = spec_meg;

fprintf('Demo V2: ISF r=%.3f vs ASF r=%.3f | EEG focal specificity unwhitened=%.3f -> shrinkage-rescued=%.3f (MEG=%.3f)\n', ...
    isf_mean_r, asf_mean_r, spec_unwhitened, spec_rescued, spec_meg);

if make_figures
    figure('Name', 'RSA-MEEG V2: ISF vs ASF and EEG Spatial Rescue');
    subplot(1, 2, 1);
    plot([1, 2], [isf_mean_r, asf_mean_r], '-o', 'LineWidth', 2.0, 'MarkerSize', 8);
    xlim([0.5, 2.5]);
    set(gca, 'XTick', [1, 2], 'XTickLabel', {'Multivariate ISF', 'Univariate ASF'});
    ylabel('Spearman r with True RDM');
    title('Zero-Mean Spatial Code Detection');

    subplot(1, 2, 2);
    plot(positions_mm(:, 1), prof_unwhitened, '--o', 'LineWidth', 1.5); hold on;
    plot(positions_mm(:, 1), prof_rescued, '-s', 'LineWidth', 2.0);
    plot(positions_mm(:, 1), prof_meg, ':d', 'LineWidth', 1.5);
    xlabel('Cortical Position (mm)');
    ylabel('Model RDM Correspondence (Spearman r)');
    legend({'EEG Unwhitened', 'EEG Shrinkage Crossnobis', 'MEG Benchmark'}, 'Location', 'best');
    title('EEG Spatial Specificity Rescue');
end
end
