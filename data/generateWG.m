function outputFile = generateWG(inputFile, outputDir)
%GENERATEWG Generate multi-view K-means initialization (*WG.mat).
%
% Based on Init.m + loadGF.m supplied by the author:
%   clu = litekmeans(Xv, c, 'MaxIter', 20);
%   Gv  = sparse(1:n, clu, ones(n,1), n, c) + 0.2;
%   datasetWG = {G1, G2, ...};
%
% USAGE:
%   generateWG('ORL.mat');
%   generateWG('Caltech101-all.mat');
%   generateWG('data/ORL.mat', 'data');
%
% OUTPUT:
%   ORL.mat               -> ORLWG.mat, variable ORLWG
%   Caltech101-all.mat    -> Caltech101allWG.mat, variable Caltech101allWG
%   small_Reuters.mat     -> smallReutersWG.mat, variable smallReutersWG
%
% Features: input MAT has X (1xV or Vx1 cell) and Y (n labels).
%           Each view can be n-by-dv or dv-by-n; both are supported.
%           Also supports 'gnd'/'labels' and 'fea' for simple datasets.
%           -v7 output can be read by scipy.io.loadmat in Python.
%
% Dependencies: prefers your existing litekmeans; otherwise MATLAB kmeans
% (Statistics and Machine Learning Toolbox); if neither is available, a
% simple local Lloyd's kmeans fallback is used (may give different labels).

    if nargin < 1 || isempty(inputFile)
        error('Specify a .mat file, e.g. generateWG(''ORL.mat'').');
    end
    inputFile = char(inputFile);
    if exist(inputFile, 'file') ~= 2
        error('Input MAT file does not exist: %s', inputFile);
    end

    [inputDir, stem, ext] = fileparts(inputFile);
    if ~strcmpi(ext, '.mat')
        error('Input must be a .mat file: %s', inputFile);
    end
    if isempty(inputDir)
        inputDir = pwd;
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = inputDir;  % Save beside input: compatible with Python clustering.py
    end
    outputDir = char(outputDir);
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end

    % Strip hyphens/underscores so file + variable names match Python's
    % existing INIT_NAMES mapping (e.g. Caltech101-all -> Caltech101allWG).
    safeStem = regexprep(stem, '[^A-Za-z0-9]', '');
    if isempty(safeStem) || ~isletter(safeStem(1))
        error('Cannot construct a valid MATLAB variable name from: %s', stem);
    end
    variableName = [safeStem, 'WG'];
    outputFile = fullfile(outputDir, [variableName, '.mat']);

    raw = load(inputFile);
    if isfield(raw, 'X')
        views = raw.X;
    elseif isfield(raw, 'fea')
        views = raw.fea;
    else
        error('%s contains neither X nor fea.', inputFile);
    end
    if ~iscell(views)
        views = {views};
    end
    views = views(:)';

    if isfield(raw, 'Y')
        labels = raw.Y;
    elseif isfield(raw, 'gnd')
        labels = raw.gnd;
    elseif isfield(raw, 'labels')
        labels = raw.labels;
    else
        error('%s contains no Y, gnd or labels; number of clusters is unknown.', inputFile);
    end
    if ~isvector(labels) || ~isnumeric(labels)
        error('Labels must be a numeric vector, such as the Y in ORL.mat.');
    end
    labels = labels(:);
    if any(~isfinite(double(labels)))
        error('Labels contain NaN/Inf.');
    end
    n = numel(labels);
    c = numel(unique(labels));
    if c < 2 || c > n
        error('Invalid number of classes: %d (n = %d).', c, n);
    end

    % Deterministic initialization across repeated runs.
    rng(40, 'twister');
    maxIter = 20;
    WG = cell(1, numel(views));
    fprintf('Input: %s | samples=%d | classes=%d | views=%d\n', ...
        inputFile, n, c, numel(views));

    for v = 1:numel(views)
        Xv = full(double(views{v}));
        if size(Xv, 1) == n
            % Already samples x features, like ORL.mat.
        elseif size(Xv, 2) == n
            Xv = Xv';
        else
            error('View %d is %dx%d; neither dimension matches n=%d.', ...
                  v, size(Xv, 1), size(Xv, 2), n);
        end
        if isempty(Xv) || any(~isfinite(Xv(:)))
            error('View %d has empty/NaN/Inf features.', v);
        end

        if exist('litekmeans', 'file') == 2
            idx = litekmeans(Xv, c, 'MaxIter', maxIter);
            backend = 'litekmeans (original)';
        elseif exist('kmeans', 'file') == 2
            idx = kmeans(Xv, c, 'MaxIter', maxIter, ...
                'Replicates', 1, 'EmptyAction', 'singleton');
            backend = 'MATLAB kmeans';
        else
            warning('No litekmeans/kmeans found; using local Lloyd kmeans fallback.');
            idx = localKmeans(Xv, c, maxIter);
            backend = 'local kmeans fallback';
        end
        idx = double(idx(:));
        if numel(idx) ~= n || any(idx < 1 | idx > c | idx ~= floor(idx))
            error('View %d produced invalid cluster assignments.', v);
        end

        % Exactly the same 0.2 + one-hot construction as Init.m.
        % This is n x c, with 1.2 at the assigned cluster and 0.2 elsewhere.
        Gv = full(sparse((1:n)', idx, ones(n, 1), n, c)) + 0.2;
        WG{v} = Gv;
        fprintf('  view %d: %d x %d -> G%d: %d x %d [%s]\n', ...
            v, size(Xv, 1), size(Xv, 2), v, n, c, backend);
    end

    % Save exactly one variable whose name matches the MAT file stem;
    % required by Python's getInitF(dataset, n_view, ...).
    saveVars = struct();
    saveVars.(variableName) = WG;
    save(outputFile, '-struct', 'saveVars', '-v7');
    fprintf('Saved: %s (variable: %s)\n', outputFile, variableName);
end

function idx = localKmeans(X, c, maxIter)
% Minimal dependency-free backup only. For exact legacy behaviour,
% put the original litekmeans.m on MATLAB's path.
    n = size(X, 1);
    centers = X(randperm(n, c), :);
    idx = zeros(n, 1);
    xNorm = sum(X.^2, 2);
    for it = 1:maxIter
        dist = bsxfun(@plus, xNorm, sum(centers.^2, 2)') - 2 * (X * centers');
        [~, newIdx] = min(dist, [], 2);
        if isequal(idx, newIdx)
            break;
        end
        idx = newIdx;
        for j = 1:c
            members = (idx == j);
            if any(members)
                centers(j, :) = mean(X(members, :), 1);
            else
                centers(j, :) = X(randi(n), :);
            end
        end
    end
end
