clear; clc;

% ==== 设置根目录 ====
ucr_root = 'D:/2025暑期科研/UCRArchive_2018/UCRArchive_2018';
dataset_dirs = dir(ucr_root);
dataset_dirs = dataset_dirs([dataset_dirs.isdir] & ~ismember({dataset_dirs.name}, {'.', '..'}));

Ks = [2, 4];         % k=2 和 k=4 分段
numInp = 1;          % 通常设为1
silent = 1;          % 静默运行

for d = 1:length(dataset_dirs)
    dataset_name = dataset_dirs(d).name;
    dataset_path = fullfile(ucr_root, dataset_name);
    fprintf('\n📂 当前数据集: %s\n', dataset_name);

    for split = ["TRAIN", "TEST"]
        file_path = fullfile(dataset_path, sprintf('%s_%s.tsv', dataset_name, split));
        if ~isfile(file_path)
            warning("❌ 未找到文件: %s", file_path); continue;
        end

        raw = readmatrix(file_path, 'FileType', 'text', 'Delimiter', '\t');
        y_all = raw(:, 1);
        X_all = raw(:, 2:end);
        sample_all = [y_all, X_all];

        valid_indices_all_k = [];
        Aout_k = containers.Map('KeyType','double','ValueType','any');
        label_k = containers.Map('KeyType','double','ValueType','any');
        valid_k = containers.Map('KeyType','double','ValueType','any');

        for k = Ks
            fprintf('  ▶️ 正在处理 %s, k = %d\n', split, k);
            Aout_all = [];
            label_all = [];
            valid_indices = [];

            for i = 1:size(X_all, 1)
                x = X_all(i, :);
                x = x(~isnan(x));
                label = y_all(i);

                % 截断为最大可被 4 整除的长度
                truncate_len = floor(length(x) / 4) * 4;
                if truncate_len == 0, continue; end
                x = x(1:truncate_len);

                if mod(length(x), k) ~= 0, continue; end

                seglen = length(x) / k;
                reshaped = reshape(x, seglen, k);
                subsample = reshaped';
                subsample = subsample';

                if rank(subsample) < size(subsample, 2), continue; end

                try
                    [Aout, ~, ~, ~, relErr] = modelEst( ...
                        'sensInd', 1:k, ...
                        'numInp',  numInp, ...
                        'data',    subsample, ...
                        'silentFlag', silent);

                    if any(isnan(Aout(:))) || any(isinf(Aout(:))) || ...
                       any(isnan(relErr)) || any(isinf(relErr))
                        continue;
                    end

                    Aout_all = [Aout_all; reshape(Aout, 1, [])];
                    label_all = [label_all; label];
                    valid_indices = [valid_indices; i];
                catch
                    continue;
                end
            end

            if isempty(Aout_all)
                warning("❌ 没有有效样本: %s_%s (k=%d)", dataset_name, split, k);
                continue;
            end

            Aout_k(k) = Aout_all;
            label_k(k) = label_all;
            valid_k(k) = valid_indices;
        end

        % ==== 统一取 k=2 和 k=4 都有效的 sample 行 ====
        if ~all(isKey(valid_k, num2cell(Ks)))
            warning("⚠️ %s_%s 中 k=2/k=4 无有效交集，跳过", dataset_name, split); continue;
        end

        common_indices = intersect(valid_k(2), valid_k(4));
        if isempty(common_indices)
            warning("⚠️ %s_%s 中无共有有效样本，跳过", dataset_name, split); continue;
        end

        [~, ia_k2] = ismember(common_indices, valid_k(2));
        [~, ia_k4] = ismember(common_indices, valid_k(4));

        Aout_data2 = Aout_k(2); label_data2 = label_k(2);
        Aout_data4 = Aout_k(4); label_data4 = label_k(4);

        Aout_final2 = Aout_data2(ia_k2, :);
        label_final2 = label_data2(ia_k2, :);
        Aout_final4 = Aout_data4(ia_k4, :);
        label_final4 = label_data4(ia_k4, :);

        % ==== 输出 Aout CSV ====
        writematrix([label_final2, Aout_final2], ...
            fullfile(dataset_path, sprintf('%s_Aout_%s_k2.csv', dataset_name, lower(split))));
        writematrix([label_final4, Aout_final4], ...
            fullfile(dataset_path, sprintf('%s_Aout_%s_k4.csv', dataset_name, lower(split))));
        fprintf('  ✅ 写入: Aout_%s_k2/k4.csv\n', lower(split));

        % ==== 输出清理后的 .tsv ====
        clean_tsv = sample_all(common_indices, :);
        tsv_path = fullfile(dataset_path, sprintf('%s_%s_cleaned.tsv', dataset_name, split));
        writematrix(clean_tsv, tsv_path, 'Delimiter','tab', 'FileType','text');
        fprintf('  ✅ 写入: Cleaned .tsv (%d 行)\n', size(clean_tsv,1));
    end
end

fprintf('\n🎉 全部数据处理完成 ✅\n');


