function out = demo_eeg_meg_spatial_rescue(make_figures)
%DEMO_EEG_MEG_SPATIAL_RESCUE Self-contained V2 demonstration of:
%   (1) Canonical 64-condition (4 clusters x 16 items, 2,016 pairs) and
%       20-condition (4x5) multivariate pattern generation
%       (rsa_meeg.sim.generate_beta_patterns / rsa.sim.generate_beta_patterns);
%   (2) 30-scale (2-60 mm) cortical Perlin spatial texture and hybrid sensor noise
%       (rsa_meeg.sim.cortical_perlin_noise);
%   (3) Multivariate Information Spread Function (ISF) vs. univariate Activation
%       Spread Function (ASF) under iso-energetic population coding; and
%   (4) Scalp EEG spatial super-resolution rescue via spatiotemporal searchlight +
%       Ledoit-Wolf shrinkage precision (rsa_meeg.rdm.shrinkage_precision) and
%       partial Spearman RDM regression (rsa_meeg.mapping.partial_compare).
%
%   Usage:
%     out = demo_eeg_meg_spatial_rescue(false);

if nargin < 1
    make_figures = true;
end
prior_rng = rng;
cleanup_rng = onCleanup(@() rng(prior_rng));
rng(20261104, 'twister');

% 1. Cortical geometry: 16 vertices along an auditory bank (5 mm spacing)
%    Focal coding patch (Heschl's gyrus / STG): vertices 6..10
%    Flanking non-coding bystander regions: vertices 1..5, 11..16
n_vertices = 16;
positions_mm = [(0:n_vertices-1).' * 5, zeros(n_vertices, 2)];
hemi = repmat({'lh'}, n_vertices, 1);
target_vertices = 6:10;
flanking_vertices = setdiff(1:n_vertices, target_vertices);
hood = rsa_meeg.geometry.neighborhoods(positions_mm, 12, 'Hemisphere', hemi);

% 2. Canonical V2 64-condition (4 clusters x 16 items, 2,016 pairs) ground-truth patterns
cluster_spec = {0.6, {0.3, 16}, {0.3, 16}, {0.3, 16}, {0.3, 16}};
n_target = numel(target_vertices);
[B_target, cov_true, rdm_true] = rsa_meeg.sim.generate_beta_patterns( ...
    cluster_spec, n_target, 'ExactCovariance', false, 'Seed', 101);

% Center each condition pattern across target vertices so univariate mean
% activation is zero (iso-energetic spatial reconfiguration code)
B_target_centered = bsxfun(@minus, B_target, mean(B_target, 2));
B_full = zeros(size(B_target, 1), n_vertices);
B_full(:, target_vertices) = 3.5 * B_target_centered;

% Update rdm_true to match the implanted zero-mean iso-energetic patterns
pairs = rdm_true.pairs;
n_pairs = size(pairs, 1);
implanted_vals = zeros(n_pairs, 1);
for p_idx = 1:n_pairs
    diff_p = B_target_centered(pairs(p_idx, 1), :) - B_target_centered(pairs(p_idx, 2), :);
    implanted_vals(p_idx) = sum(diff_p.^2) / n_target;
end
rdm_true.values = implanted_vals;

% Construct a nuisance covariate RDM (e.g., low-level stimulus amplitude)
n_cond = size(B_target, 1);
nuisance_vec = sin((1:n_cond).' * 0.7);
nuisance_model = struct();
nuisance_model.condition_ids = 1:n_cond;
nuisance_model.values = abs(nuisance_vec(pairs(:, 1)) - nuisance_vec(pairs(:, 2)));

% 3. Forward/inverse spatial resolution operators:
%    - MEG Gradiometer benchmark: focal spatial resolution (sigma = 5 mm)
%    - EEG Volume Conduction: broad spatial blurring (sigma = 18 mm) + strong
%      common-mode crosstalk across neighboring vertices
dist_mat = abs(bsxfun(@minus, positions_mm(:, 1), positions_mm(:, 1).'));
R_meg = exp(-0.5 * (dist_mat / 5.0).^2);
R_meg = bsxfun(@rdivide, R_meg, sum(R_meg, 2));

R_eeg_focal = exp(-0.5 * (dist_mat / 6.0).^2);
R_eeg_focal = bsxfun(@rdivide, R_eeg_focal, sum(R_eeg_focal, 2));
R_eeg_blur = exp(-0.5 * (dist_mat / 22.0).^2);
R_eeg_blur = bsxfun(@rdivide, R_eeg_blur, sum(R_eeg_blur, 2));
R_eeg = 0.28 * R_eeg_focal + 0.72 * R_eeg_blur;

% 4. Temporal profiles (7 samples, 5 ms step): distinct micro-temporal phase
%    evolution within the target patch vs. shared zero-lag volume-conducted background
times_s = (0:6) * 0.005;
n_times = numel(times_s);
temp_basis = zeros(n_vertices, n_times);
for v = 1:n_vertices
    phase_shift = 0.35 * (v - 8);
    temp_basis(v, :) = exp(-((times_s - 0.015) / 0.012).^2) .* cos(2 * pi * 40 * times_s + phase_shift);
end

% 5. Simulate multi-subject (N = 8, 2 folds x 64 conditions = 128 observations)
n_subjects = 8;
n_folds = 2;
condition_ids = repmat((1:n_cond).', n_folds, 1);
fold_ids = repelem((1:n_folds).', n_cond);
n_obs = numel(condition_ids);

maps_meg_isf = zeros(n_subjects, n_vertices);
maps_eeg_unwhitened = zeros(n_subjects, n_vertices);
maps_eeg_rescued = zeros(n_subjects, n_vertices);
maps_eeg_partial = zeros(n_subjects, n_vertices);
maps_univariate_asf = zeros(n_subjects, n_vertices);

for s = 1:n_subjects
    % Clean cortical space-time patterns before volume conduction
    X_clean = zeros(n_obs, n_vertices, n_times);
    for o = 1:n_obs
        c = condition_ids(o);
        X_clean(o, :, :) = bsxfun(@times, B_full(c, :).', temp_basis);
    end

    % 30-scale cortical Perlin background noise
    [perlin_bg, ~] = rsa_meeg.sim.cortical_perlin_noise( ...
        positions_mm, n_obs, n_times, 'ScalesMm', linspace(2, 60, 30), ...
        'SpectralSlope', 1.0, 'Hemisphere', hemi, 'Seed', 1000 + s);

    % Common-mode volume-conducted background noise in EEG
    common_mode = randn(n_obs, 1, n_times);
    spatial_smear = reshape(exp(-0.5 * ((positions_mm(:, 1) - 35) / 28.0).^2), 1, n_vertices, 1);
    eeg_bg = 1.4 * bsxfun(@times, common_mode, spatial_smear) + 0.20 * perlin_bg;
    meg_bg = 0.20 * perlin_bg;

    X_meg = zeros(n_obs, n_vertices, n_times);
    X_eeg = zeros(n_obs, n_vertices, n_times);
    for t = 1:n_times
        X_meg(:, :, t) = X_clean(:, :, t) * R_meg.' + meg_bg(:, :, t);
        X_eeg(:, :, t) = X_clean(:, :, t) * R_eeg.' + eeg_bg(:, :, t);
    end

    ds_meg = rsa_meeg.dataset(X_meg, times_s, condition_ids, 'FoldIds', fold_ids, ...
        'FeatureIds', 1:n_vertices, 'FeatureTypes', repmat({'source'}, n_vertices, 1), ...
        'Positions', positions_mm, 'Hemisphere', hemi, 'SubjectId', s);
    ds_eeg = rsa_meeg.dataset(X_eeg, times_s, condition_ids, 'FoldIds', fold_ids, ...
        'FeatureIds', 1:n_vertices, 'FeatureTypes', repmat({'source'}, n_vertices, 1), ...
        'Positions', positions_mm, 'Hemisphere', hemi, 'SubjectId', s);

    % (a) MEG ISF
    P_meg = rsa_meeg.rdm.shrinkage_precision(ds_meg, hood, ...
        'Method', 'ledoit-wolf', 'Mode', 'spatiotemporal', 'WindowSamples', n_times);
    R_meg_rdm = rsa_meeg.rdm.searchlight(ds_meg, hood, ...
        'WindowSamples', n_times, 'StepSamples', n_times, ...
        'Mode', 'spatiotemporal', 'Metric', 'crossnobis', 'Precision', P_meg);
    fit_meg = rsa_meeg.mapping.compare(R_meg_rdm, rdm_true, 'Correlation', 'spearman');
    maps_meg_isf(s, :) = fit_meg.values(:, 1).';

    % (b) Unwhitened spatial EEG RSA (conventional spatial average without shrinkage whitening)
    R_eeg_unw = rsa_meeg.rdm.searchlight(ds_eeg, hood, ...
        'WindowSamples', n_times, 'StepSamples', n_times, ...
        'Mode', 'spatial', 'Metric', 'sqeuclidean');
    fit_eeg_unw = rsa_meeg.mapping.compare(R_eeg_unw, rdm_true, 'Correlation', 'spearman');
    maps_eeg_unwhitened(s, :) = fit_eeg_unw.values(:, 1).';

    % (c) Rescued EEG RSA: Spatiotemporal searchlight + Ledoit-Wolf shrinkage precision
    P_eeg = rsa_meeg.rdm.shrinkage_precision(ds_eeg, hood, ...
        'Method', 'ledoit-wolf', 'Mode', 'spatiotemporal', 'WindowSamples', n_times);
    R_eeg_res = rsa_meeg.rdm.searchlight(ds_eeg, hood, ...
        'WindowSamples', n_times, 'StepSamples', n_times, ...
        'Mode', 'spatiotemporal', 'Metric', 'crossnobis', 'Precision', P_eeg);
    fit_eeg_res = rsa_meeg.mapping.compare(R_eeg_res, rdm_true, 'Correlation', 'spearman');
    maps_eeg_rescued(s, :) = fit_eeg_res.values(:, 1).';

    % (d) Partial Spearman RDM correlation controlling for nuisance RDM
    fit_eeg_part = rsa_meeg.mapping.partial_compare(R_eeg_res, rdm_true, nuisance_model, ...
        'Correlation', 'spearman');
    maps_eeg_partial(s, :) = fit_eeg_part.values(:, 1).';

    % (e) Univariate ASF (scalar bulk mean activation across neighborhood features & time)
    X_meg_bulk = mean(X_meg, 3);
    ds_meg_bulk = rsa_meeg.dataset(X_meg_bulk, mean(times_s), condition_ids, 'FoldIds', fold_ids, ...
        'FeatureIds', 1:n_vertices, 'FeatureTypes', repmat({'source'}, n_vertices, 1), ...
        'Positions', positions_mm, 'Hemisphere', hemi, 'SubjectId', s);
    R_asf = rsa_meeg.rdm.searchlight(ds_meg_bulk, hood, ...
        'WindowSamples', 1, 'StepSamples', 1, ...
        'Mode', 'temporal', 'Metric', 'sqeuclidean');
    fit_asf = rsa_meeg.mapping.compare(R_asf, rdm_true, 'Correlation', 'spearman');
    maps_univariate_asf(s, :) = fit_asf.values(:, 1).';
end

% 6. Summary metrics: Focal-to-Flanking Specificity Ratio & Target Recovery
mean_meg_isf = mean(maps_meg_isf, 1);
mean_eeg_unw = mean(maps_eeg_unwhitened, 1);
mean_eeg_res = mean(maps_eeg_rescued, 1);
mean_eeg_part = mean(maps_eeg_partial, 1);
mean_asf = mean(maps_univariate_asf, 1);

focal_gain_unw = mean(mean_eeg_unw(target_vertices)) - mean(mean_eeg_unw(flanking_vertices));
focal_gain_res = mean(mean_eeg_res(target_vertices)) - mean(mean_eeg_res(flanking_vertices));

out = struct();
out.positions_mm = positions_mm(:, 1);
out.target_vertices = target_vertices;
out.flanking_vertices = flanking_vertices;
out.cov_true = cov_true;
out.rdm_true = rdm_true;
out.maps_meg_isf = maps_meg_isf;
out.maps_eeg_unwhitened = maps_eeg_unwhitened;
out.maps_eeg_rescued = maps_eeg_rescued;
out.maps_eeg_partial = maps_eeg_partial;
out.maps_univariate_asf = maps_univariate_asf;
out.mean_meg_isf = mean_meg_isf;
out.mean_eeg_unwhitened = mean_eeg_unw;
out.mean_eeg_rescued = mean_eeg_res;
out.mean_eeg_partial = mean_eeg_part;
out.mean_univariate_asf = mean_asf;
out.focal_specificity_eeg_unwhitened = focal_gain_unw;
out.focal_specificity_eeg_rescued = focal_gain_res;
out.focal_specificity_unwhitened = focal_gain_unw;
out.focal_specificity_rescued = focal_gain_res;
out.isf_mean_r = mean(mean_meg_isf(target_vertices));
out.asf_mean_r = mean(mean_asf(target_vertices));

fprintf(['V2 Spatial Rescue Demo (64 conditions, 2,016 pairs):\n', ...
    '  MEG ISF target r = %.3f | Univariate ASF target r = %.3f\n', ...
    '  EEG Unwhitened focal specificity = %.3f -> Rescued (Shrinkage + ST) = %.3f\n'], ...
    mean(mean_meg_isf(target_vertices)), mean(mean_asf(target_vertices)), ...
    focal_gain_unw, focal_gain_res);

if make_figures
    figure('Name', 'RSA-MEEG V2: ISF vs ASF & EEG Spatial Rescue', 'Color', 'w');
    subplot(1, 2, 1);
    imagesc(rdm_true.square);
    axis square; colorbar;
    title('64-Condition Ground-Truth RDM (4x16)');
    xlabel('Condition'); ylabel('Condition');

    subplot(1, 2, 2);
    plot(positions_mm(:, 1), mean_meg_isf, '-o', 'LineWidth', 1.8); hold on;
    plot(positions_mm(:, 1), mean_eeg_res, '-s', 'LineWidth', 1.8);
    plot(positions_mm(:, 1), mean_eeg_unw, '--^', 'LineWidth', 1.4);
    plot(positions_mm(:, 1), mean_asf, ':d', 'LineWidth', 1.4);
    xline(positions_mm(target_vertices(1), 1), 'k--');
    xline(positions_mm(target_vertices(end), 1), 'k--');
    legend({'MEG ISF (Shrinkage)', 'EEG Rescued (ST + Shrinkage)', ...
        'EEG Unwhitened (Spatial)', 'Univariate ASF'}, 'Location', 'best');
    xlabel('Cortical Coordinate (mm)');
    ylabel('Model Spearman r');
    title('Cortical Information Spread Function (ISF)');
    grid on;
end
end
