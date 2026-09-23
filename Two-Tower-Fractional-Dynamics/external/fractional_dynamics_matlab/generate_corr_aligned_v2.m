%% generate_corr_aligned_v2.m
% Generate k=2 correlation-matrix features aligned exactly with the
% Aout/cleaned-TSV sample universe used by the unified V2 experiments.
%
% Scientific invariants enforced by this script:
%   1. Read *_TRAIN_cleaned.tsv and *_TEST_cleaned.tsv, never the raw TSV.
%   2. Verify each cleaned TSV has the same row count and labels as Aout k=2.
%   3. Use the same multiple-of-4 truncation used by run_ucr_k2k4all.m.
%   4. Produce exactly one correlation row for every cleaned TSV row.
%   5. Never silently skip a sample.
%   6. Write new CorrAligned files so legacy Corr files are not overwritten.
%
% Output per dataset and split:
%   <dataset>_CorrAligned_train_k2.csv
%   <dataset>_CorrAligned_test_k2.csv
%   <dataset>_CorrAligned_train_k2_audit.csv
%   <dataset>_CorrAligned_test_k2_audit.csv
%
% The feature CSV layout is unchanged from the legacy correlation files:
%   column 1 = class label
%   columns 2:5 = reshape([1 r; r 1], 1, [])

clear; clc;

% -------------------------------------------------------------------------
% EDIT THIS PATH IF NEEDED
% -------------------------------------------------------------------------
UCR_ROOT = 'D:/2025暑期科研/UCRArchive_2018/UCRArchive_2018';

EXPECTED_DATASET_COUNT = 125;
TRUNCATION_MULTIPLE = 4;   % Must match run_ucr_k2k4all.m
ZERO_VARIANCE_TOL = 1e-12;
LABEL_TOL = 1e-10;

if ~isfolder(UCR_ROOT)
    error('UCR root does not exist: %s', UCR_ROOT);
end

dataset_dirs = dir(UCR_ROOT);
dataset_dirs = dataset_dirs([dataset_dirs.isdir]);
dataset_dirs = dataset_dirs(~ismember({dataset_dirs.name}, {'.', '..'}));
dataset_dirs = dataset_dirs(~startsWith({dataset_dirs.name}, '_'));

% dataset, split, status, samples, zero-variance fallbacks, output, message
summary_rows = cell(0, 7);
datasets_attempted = 0;

fprintf('Generating aligned k=2 correlation features from cleaned TSV files.\n');
fprintf('UCR root: %s\n', UCR_ROOT);

for d = 1:numel(dataset_dirs)
    dataset_name = dataset_dirs(d).name;
    dataset_path = fullfile(UCR_ROOT, dataset_name);

    train_cleaned = fullfile(dataset_path, ...
        sprintf('%s_TRAIN_cleaned.tsv', dataset_name));
    test_cleaned = fullfile(dataset_path, ...
        sprintf('%s_TEST_cleaned.tsv', dataset_name));

    % Ignore non-dataset folders that may exist under UCR_ROOT.
    if ~isfile(train_cleaned) && ~isfile(test_cleaned)
        continue;
    end

    datasets_attempted = datasets_attempted + 1;
    fprintf('\n[%d] Dataset: %s\n', datasets_attempted, dataset_name);

    for split = ["TRAIN", "TEST"]
        split_name = char(split);
        split_lower = lower(split_name);

        cleaned_path = fullfile(dataset_path, ...
            sprintf('%s_%s_cleaned.tsv', dataset_name, split_name));
        aout_path = fullfile(dataset_path, ...
            sprintf('%s_Aout_%s_k2.csv', dataset_name, split_lower));
        output_path = fullfile(dataset_path, ...
            sprintf('%s_CorrAligned_%s_k2.csv', dataset_name, split_lower));
        audit_path = fullfile(dataset_path, ...
            sprintf('%s_CorrAligned_%s_k2_audit.csv', dataset_name, split_lower));

        try
            if ~isfile(cleaned_path)
                error('Missing cleaned TSV: %s', cleaned_path);
            end
            if ~isfile(aout_path)
                error('Missing Aout k=2 file required for alignment audit: %s', aout_path);
            end

            cleaned = readmatrix(cleaned_path, ...
                'FileType', 'text', 'Delimiter', '\t');
            aout = readmatrix(aout_path);

            if isempty(cleaned) || size(cleaned, 2) < 2
                error('Cleaned TSV is empty or has no time-series columns.');
            end
            if isempty(aout) || size(aout, 2) < 2
                error('Aout file is empty or malformed.');
            end

            labels = cleaned(:, 1);
            X_all = cleaned(:, 2:end);
            aout_labels = aout(:, 1);
            n_samples = size(cleaned, 1);

            if size(aout, 1) ~= n_samples
                error(['Aout/cleaned row mismatch: cleaned=%d, Aout=%d. ' ...
                       'Do not generate Corr until this provenance issue is resolved.'], ...
                       n_samples, size(aout, 1));
            end
            if ~labels_equal(labels, aout_labels, LABEL_TOL)
                error(['Aout labels are not aligned with cleaned TSV labels. ' ...
                       'Row order cannot be trusted.']);
            end

            corr_features = nan(n_samples, 4);
            effective_length = zeros(n_samples, 1);
            truncated_length = zeros(n_samples, 1);
            correlation_r = zeros(n_samples, 1);
            zero_variance_fallback = false(n_samples, 1);

            for i = 1:n_samples
                x = X_all(i, :);
                x = x(~isnan(x));

                if isempty(x) || any(~isfinite(x))
                    error('Row %d contains no finite series or contains Inf.', i);
                end

                effective_length(i) = numel(x);
                L = floor(numel(x) / TRUNCATION_MULTIPLE) * TRUNCATION_MULTIPLE;
                truncated_length(i) = L;

                if L < TRUNCATION_MULTIPLE
                    error('Row %d is too short after multiple-of-4 truncation (L=%d).', i, L);
                end

                % Match run_ucr_k2k4all.m exactly: first truncate to a
                % multiple of 4, then split the retained series into k=2
                % contiguous segments of equal length.
                x = x(1:L);
                segment_length = L / 2;
                seg1 = x(1:segment_length);
                seg2 = x(segment_length + 1:end);

                centered1 = seg1 - mean(seg1);
                centered2 = seg2 - mean(seg2);
                denom = sqrt(sum(centered1 .^ 2) * sum(centered2 .^ 2));
                denom_scale = max(1.0, norm(seg1) * norm(seg2));

                if denom <= ZERO_VARIANCE_TOL * denom_scale
                    % Pearson correlation is undefined if either segment
                    % has zero variance. Use the documented neutral value
                    % r=0 while retaining the sample and record the event.
                    r = 0.0;
                    zero_variance_fallback(i) = true;
                else
                    r = sum(centered1 .* centered2) / denom;
                    % Protect against tiny floating-point excursions.
                    r = max(-1.0, min(1.0, r));
                end

                C = [1.0, r; r, 1.0];
                corr_features(i, :) = reshape(C, 1, []);
                correlation_r(i) = r;
            end

            if size(corr_features, 1) ~= n_samples
                error('Internal error: output/sample row counts differ.');
            end
            if any(~isfinite(corr_features(:)))
                error('Internal error: generated CorrAligned features contain NaN/Inf.');
            end

            output = [labels, corr_features];
            if ~labels_equal(output(:, 1), labels, LABEL_TOL)
                error('Internal error: output labels changed during generation.');
            end

            audit = table( ...
                (1:n_samples)', ...
                labels, ...
                effective_length, ...
                truncated_length, ...
                correlation_r, ...
                zero_variance_fallback, ...
                'VariableNames', { ...
                    'cleaned_row', ...
                    'label', ...
                    'effective_length', ...
                    'truncated_length', ...
                    'correlation_r', ...
                    'zero_variance_fallback'} ...
            );

            atomic_write_matrix(output, output_path);
            atomic_write_table(audit, audit_path);

            fallback_count = sum(zero_variance_fallback);
            fprintf('  %s: saved %d aligned samples, zero-variance fallback=%d\n', ...
                split_name, n_samples, fallback_count);

            summary_rows(end + 1, :) = { ...
                dataset_name, split_name, 'completed', n_samples, ...
                fallback_count, output_path, ''}; %#ok<SAGROW>

        catch ME
            fprintf(2, '  %s FAILED: %s\n', split_name, ME.message);
            summary_rows(end + 1, :) = { ...
                dataset_name, split_name, 'failed', 0, 0, ...
                output_path, ME.message}; %#ok<SAGROW>
        end
    end
end

summary_table = cell2table(summary_rows, ...
    'VariableNames', { ...
        'dataset', ...
        'split', ...
        'status', ...
        'samples_written', ...
        'zero_variance_fallbacks', ...
        'output_path', ...
        'message'} ...
);

summary_path = fullfile(UCR_ROOT, 'CorrAlignedV2_generation_summary.csv');
atomic_write_table(summary_table, summary_path);

completed_splits = sum(strcmp(summary_table.status, 'completed'));
failed_splits = sum(strcmp(summary_table.status, 'failed'));

fprintf('\n============================================================\n');
fprintf('Correlation V2 generation finished.\n');
fprintf('Dataset folders attempted: %d\n', datasets_attempted);
fprintf('Completed splits: %d\n', completed_splits);
fprintf('Failed splits: %d\n', failed_splits);
fprintf('Summary: %s\n', summary_path);

if datasets_attempted ~= EXPECTED_DATASET_COUNT
    warning('Expected %d datasets but discovered %d cleaned dataset folders.', ...
        EXPECTED_DATASET_COUNT, datasets_attempted);
end
if failed_splits > 0
    warning('%d TRAIN/TEST splits failed. Inspect the generation summary before training.', ...
        failed_splits);
else
    fprintf('All discovered TRAIN/TEST splits passed row and label alignment checks.\n');
end


%% Local helper functions
function tf = labels_equal(a, b, tolerance)
    a = double(a(:));
    b = double(b(:));
    tf = numel(a) == numel(b) && ...
         all(isfinite(a)) && all(isfinite(b)) && ...
         all(abs(a - b) <= tolerance);
end


function atomic_write_matrix(values, final_path)
    output_dir = fileparts(final_path);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    temp_path = [tempname(output_dir), '.csv'];
    cleanup = onCleanup(@() cleanup_temp(temp_path));
    writematrix(values, temp_path);
    [ok, message] = movefile(temp_path, final_path, 'f');
    if ~ok
        error('Could not publish %s: %s', final_path, message);
    end
    clear cleanup;
end


function atomic_write_table(values, final_path)
    output_dir = fileparts(final_path);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    temp_path = [tempname(output_dir), '.csv'];
    cleanup = onCleanup(@() cleanup_temp(temp_path));
    writetable(values, temp_path);
    [ok, message] = movefile(temp_path, final_path, 'f');
    if ~ok
        error('Could not publish %s: %s', final_path, message);
    end
    clear cleanup;
end


function cleanup_temp(path_value)
    if isfile(path_value)
        delete(path_value);
    end
end
