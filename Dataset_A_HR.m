% This MATLAB script can be used to reproduce the hemodynamic response in dataset A (Figure 6)
% Please download BBCItoolbox to 'MyToolboxDir'
% Please download dataset to 'NirsMyDataDir' and 'EegMyDataDir'
% The authors would be grateful if published reports of research using this code
% (or a modified version, maintaining a significant portion of the original code) would cite the following article:
% Shin et al. "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset",
% Scientific data (2017), under review.

%% Plot figures
% This MATLAB script can be used to reproduce the hemodynamic response in dataset A (Figure 3)
% Please download BBCItoolbox to 'MyToolboxDir'
% Please download dataset to 'NirsMyDataDir' and 'EegMyDataDir'
% The authors would be grateful if published reports of research using this code
% (or a modified version, maintaining a significant portion of the original code) would cite the following article:
% Shin et al. "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset",
% Scientific data (2017), under review.

clear all; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%% modify directory paths properly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MyToolboxDir = fullfile('C:','Users','shin','Documents','MATLAB','bbci_public-master');
MyToolboxDir = fullfile('/NAS','home','kh_guy','Capstone','bbci_public');
WorkingDir = fullfile('/NAS','home','kh_guy','Capstone','Capstone_490');
NirsMyDataDir = fullfile('/NAS','home','kh_guy','Capstone','ProcessedShinData','Shin_Data','NIRS_01-26_MATLAB');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',NirsMyDataDir,'TmpDir','/tmp/','History',0);
cd(WorkingDir);

addpath(genpath(pwd));
load -ascii 'cmap_asymmetry.mat';

%%%%%%%%%%%%%%%%% initial parameter %%%%%%%%%%%%%%%%%
subdir_list = {'VP001-NIRS','VP002-NIRS','VP003-NIRS','VP004-NIRS','VP005-NIRS','VP006-NIRS','VP007-NIRS','VP008-NIRS','VP009-NIRS','VP010-NIRS','VP011-NIRS','VP012-NIRS','VP013-NIRS','VP014-NIRS','VP015-NIRS','VP016-NIRS','VP017-NIRS','VP018-NIRS','VP019-NIRS','VP020-NIRS','VP021-NIRS','VP022-NIRS','VP023-NIRS','VP024-NIRS','VP025-NIRS','VP026-NIRS'};
%subdir_list = {'VP001-NIRS','VP002-NIRS','VP003-NIRS'};
%subdir_list = {'VP001-NIRS'};
ival_epo  = [-5 60]*1000; % epoch range (unit: msec)
ival_base = [-5 -2]*1000; % baseline correction range (unit: msec)
ival_scalp = [0 10; 10 20; 20 30; 30 40; 40 60]; % (unit: sec)
ylim = [-2 2]*1e-3;
classDef= {7, 8, 9; '0-back','2-back','3-back'}; % redefine name according to the text
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load NIRS data
for vp = 1 : length(subdir_list)
    
    disp([subdir_list{vp}, ' was started']);
    loadDir = fullfile(NirsMyDataDir,subdir_list{vp});
    cd(loadDir);
    load cnt_nback; load mrk_nback; load mnt_nback;
    cd(WorkingDir);
    
    % redefine the name of class
    mrk_nback = mrk_defineClasses(mrk_nback, classDef);
    
    % low-pass filter
    [b,a] = butter(3, 0.2/cnt_nback.deoxy.fs*2, 'low');
    cnt_nback.deoxy = proc_filtfilt(cnt_nback.deoxy, b, a);
    cnt_nback.oxy   = proc_filtfilt(cnt_nback.oxy, b, a);
    
    % segmentation
    epo.deoxy = proc_segmentation(cnt_nback.deoxy, mrk_nback, ival_epo);
    epo.oxy   = proc_segmentation(cnt_nback.oxy,   mrk_nback, ival_epo);
    
    % Add unit of x- and y-axis
    epo.deoxy.xUnit = 's';
    epo.deoxy.yUnit = 'mmol/L';
    epo.oxy.xUnit = 's';
    epo.oxy.yUnit = 'mmol/L';
	
	%% Baseline correction
	epo.deoxy = proc_baseline(epo.deoxy, ival_base);
	epo.oxy   = proc_baseline(epo.oxy, ival_base);

	%% Dimensionality correction
	epo.deoxy.t = epo.deoxy.t(:,:,1);
	epo.oxy.t = epo.oxy.t(:,:,1);

	% msec -> sec
	epo.deoxy.t = epo.deoxy.t/1000;
	epo.oxy.t   = epo.oxy.t/1000;

	epo.deoxy.refIval = ival_base/1000;
	epo.oxy.refIval = ival_base/1000;
	%% Trial-Average
	epo.deoxy = proc_average(epo.deoxy, 'Stats', 1);
	epo.oxy   = proc_average(epo.oxy, 'Stats', 1);
	
	subplot(27,4,(vp-1)*4 + 1);
	plot_channel(epo.deoxy, 'AF7');
	subplot(27,4,(vp-1)*4 + 2);
	plot_channel(epo.deoxy, 'C3h');
	subplot(27,4,(vp-1)*4 + 3);
	plot_channel(epo.oxy, 'AF7');
	subplot(27,4,(vp-1)*4 + 4);
	plot_channel(epo.oxy, 'C3h');
	
    if vp == 1
        epo_all.deoxy = epo.deoxy;
        epo_all.oxy   = epo.oxy;
    else
        epo_all.deoxy = proc_appendEpochs(epo_all.deoxy, epo.deoxy);
        epo_all.oxy   = proc_appendEpochs(epo_all.oxy,   epo.oxy);
    end
end

%% Baseline correction
epo_all.deoxy = proc_baseline(epo_all.deoxy, ival_base);
epo_all.oxy   = proc_baseline(epo_all.oxy, ival_base);

%% Dimensionality correction
epo_all.deoxy.t = epo_all.deoxy.t(:,:,1);
epo_all.oxy.t = epo_all.oxy.t(:,:,1);

% msec -> sec
epo_all.deoxy.t = epo_all.deoxy.t/1000;
epo_all.oxy.t   = epo_all.oxy.t/1000;

epo_all.deoxy.refIval = ival_base/1000;
epo_all.oxy.refIval = ival_base/1000;
%% Trial-Average
epo_all.deoxy = proc_average(epo_all.deoxy, 'Stats', 1);
epo_all.oxy   = proc_average(epo_all.oxy, 'Stats', 1);

%% Plot figures
subplot(27,4,1);
plot_channel(epo_all.deoxy, 'AF7');
subplot(27,4,2);
plot_channel(epo_all.deoxy, 'C3h');
subplot(27,4,3);
plot_channel(epo_all.oxy, 'AF7');
subplot(27,4,4);
plot_channel(epo_all.oxy, 'C3h');

