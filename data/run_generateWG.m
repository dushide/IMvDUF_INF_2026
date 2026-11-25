%% One-click batch generator for *WG.mat files
% Open this script in MATLAB and press Run. No command-line arguments.
clearvars;
clc;

% Location of ORL.mat / Caltech101-all.mat / etc.
% By default, put the datasets in the same folder as these .m files.
DATA_DIR = fileparts(mfilename('fullpath'));
% If your data lives in the Python project, replace the line above with:
% DATA_DIR = '/data/dusd/code/Source Code/data';

% Add the folder containing your original litekmeans.m if available.
FUNCTIONS_DIR = fullfile(fileparts(mfilename('fullpath')), 'functions');
if exist(FUNCTIONS_DIR, 'dir') == 7
    addpath(genpath(FUNCTIONS_DIR));
end

% Edit this list freely. Only files listed here will be processed.
DATASETS = {
    'esp-game.mat'
    'flickr.mat'
    % 'Caltech101-all.mat'
    % 'MNIST10k.mat'
    % 'NUS-WIDE.mat'
    % 'MITIndoor.mat'
    % 'scene15.mat'
    % 'small_Reuters.mat'
};

numSuccess = 0;
for i = 1:numel(DATASETS)
    inputFile = fullfile(DATA_DIR, DATASETS{i});
    fprintf('\n==== %d/%d: %s ====\n', i, numel(DATASETS), DATASETS{i});
    try
        outputFile = generateWG(inputFile, DATA_DIR);
        fprintf('[OK] %s\n', outputFile);
        numSuccess = numSuccess + 1;
    catch ME
        fprintf(2, '[FAILED] %s: %s\n', DATASETS{i}, ME.message);
    end
end
fprintf('\nFinished: %d/%d dataset(s) generated successfully.\n', ...
    numSuccess, numel(DATASETS));
