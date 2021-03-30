function varargout = process_NIRSpreprocess( varargin )

% @=============================================================================
% This function is part of the Brainstorm software:
% http://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2016 University of Southern California & McGill University
% This software is distributed under the terms of the GNU General Public License
% as published by the Free Software Foundation. Further details on the GPLv3
% license can be found at http://www.gnu.org/copyleft/gpl.html.
% 
% FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE
% UNIVERSITY OF SOUTHERN CALIFORNIA AND ITS COLLABORATORS DO NOT MAKE ANY
% WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY
% LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
%
% For more information type "brainstorm license" at command prompt.
% =============================================================================@
%
% Authors: Edouard Delaire, 2020; Thomas Vincent, 2015-2019


eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    %TOCHECK: how do we limit the input file types (only NIRS data)?
    sProcess.Comment     = 'Big Process for Capstone 490';
    sProcess.FileTag     = '';
    sProcess.Category    = 'File';
    sProcess.SubGroup    = 'Capstone';
    sProcess.Index       = 4000; %0: not shown, >0: defines place in the list of processes
    sProcess.Description = 'https://github.com/NK2306/Capstone_490';
    sProcess.isSeparator = 0; 
    
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data', 'raw'};
    % Definition of the outputs of this process
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 0;
    
    %Definition of the options
    
    %////////////////////////////////////////////////////////// Motion Correction /////////////////////////////////////////////////////
    sProcess.options.option_motion_correction.Comment = '<b>Motion Correction</b>';
    sProcess.options.option_motion_correction.Type    = 'checkbox';
    sProcess.options.option_motion_correction.Value   = 0;
    sProcess.options.option_motion_correction.Controller='motion_correction';
    
    %////////////////////////////////////////////////////////// NIRS /////////////////////////////////////////////////////
    sProcess.options.option_NIRS_preprocessing.Comment = '<b>NIRS preprocessing</b>';
    sProcess.options.option_NIRS_preprocessing.Type    = 'checkbox';
    sProcess.options.option_NIRS_preprocessing.Value   = 0;
    sProcess.options.option_NIRS_preprocessing.Controller='NIRS';
    
    sProcess.options.option_NIRS_number_of_participants.Comment = 'Number of participants: ';
    sProcess.options.option_NIRS_number_of_participants.Type    = 'value';
    sProcess.options.option_NIRS_number_of_participants.Value   = {26, '', 0};
    sProcess.options.option_NIRS_number_of_participants.Class='NIRS';
    
    sProcess.options.option_NIRS_number_of_trials_per_task.Comment = 'Number of trials for each n-back task: ';
    sProcess.options.option_NIRS_number_of_trials_per_task.Type    = 'value';
    sProcess.options.option_NIRS_number_of_trials_per_task.Value   = {9, '', 0};
    sProcess.options.option_NIRS_number_of_trials_per_task.Class='NIRS';
    
    sProcess.options.option_apply_LPF.Comment = 'Low-pass filter';
    sProcess.options.option_apply_LPF.Type    = 'checkbox';
    sProcess.options.option_apply_LPF.Value   = 0;
    sProcess.options.option_apply_LPF.Controller='NIRS-LPF';
    
    sProcess.options.option_keep_mean.Comment = 'Keep mean';
    sProcess.options.option_keep_mean.Type    = 'checkbox';
    sProcess.options.option_keep_mean.Value   = 1;
    sProcess.options.option_keep_mean.Class='NIRS-LPF';
    
    sProcess.options.option_low_cutoff.Comment = 'Lower cutoff frequency (0=disable):';
    sProcess.options.option_low_cutoff.Type    = 'value';
    sProcess.options.option_low_cutoff.Value   = {0.01, '', 4};
    sProcess.options.option_low_cutoff.Class='NIRS-LPF';
    
    sProcess.options.option_high_cutoff.Comment = 'Upper cutoff frequency (0=disable):';
    sProcess.options.option_high_cutoff.Type    = 'value';
    sProcess.options.option_high_cutoff.Value   = {0.5, '', 4};
    sProcess.options.option_high_cutoff.Class='NIRS-LPF';
    
    sProcess.options.order.Comment = 'Filter order:';
    sProcess.options.order.Type    = 'value';
    sProcess.options.order.Value   = {3, '', 0};
    sProcess.options.order.Class='NIRS-LPF';
    
    sProcess.options.bad_trials_removal.Comment = 'Bad trial removal';
    sProcess.options.bad_trials_removal.Type    = 'checkbox';
    sProcess.options.bad_trials_removal.Value   = 0;
    sProcess.options.bad_trials_removal.Class='NIRS';
    sProcess.options.bad_trials_removal.Controller='variation';
    
    sProcess.options.coefficient_variation.Comment = 'Trial rejection by coefficient variation';
    sProcess.options.coefficient_variation.Type    = 'value';
    sProcess.options.coefficient_variation.Value   = {10, '%', 0};
    sProcess.options.coefficient_variation.Class='variation';
    
    sProcess.options.bad_channels_percentage.Comment = 'Percentage of bad channels to disqualify a trial:  ';
    sProcess.options.bad_channels_percentage.Type    = 'value';
    sProcess.options.bad_channels_percentage.Value   = {60, '%', 0};
    sProcess.options.bad_channels_percentage.Class='variation';
    
    sProcess.options.option_NIRS_average.Comment = 'Generate average files<BR><BR>';
    sProcess.options.option_NIRS_average.Type    = 'checkbox';
    sProcess.options.option_NIRS_average.Value   = 0;
    sProcess.options.option_NIRS_average.Class='NIRS';
    
    %////////////////////////////////////////////////////////// EEG /////////////////////////////////////////////////////
    sProcess.options.option_EEG_preprocessing.Comment = '<b>EEG preprocessing</b>';
    sProcess.options.option_EEG_preprocessing.Type    = 'checkbox';
    sProcess.options.option_EEG_preprocessing.Value   = 0;
    sProcess.options.option_EEG_preprocessing.Controller='EEG';
    
    
    %////////////////////////////////////////////////////////// Machine Learning /////////////////////////////////////////////////////
    
    sProcess.options.option_ML_classifier.Comment = '<b>Train ML classifier</b>';
    sProcess.options.option_ML_classifier.Type    = 'checkbox';
    sProcess.options.option_ML_classifier.Value   = 0;
    sProcess.options.option_ML_classifier.Controller='ML';
    
    sProcess.options.option_data_type.Comment = 'Type of Data (EEG or NIRS)';
    sProcess.options.option_data_type.Type = 'text';
    sProcess.options.option_data_type.Value = 'EEG';
    sProcess.options.option_data_type.Class='ML';
    
    sProcess.options.restrict_data.Comment = 'Restrict Data Size (True or False)';
    sProcess.options.restrict_data.Type = 'text';
    sProcess.options.restrict_data.Value = 'False';
    sProcess.options.restrict_data.Class='ML';
end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end

%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>

    OutputFiles = {};
    sFiles = sInputs.FileName;
    current_protocol = bst_get('ProtocolInfo');
    num_of_participants = sProcess.options.option_NIRS_number_of_participants.Value{1};
    num_of_trials = sProcess.options.option_NIRS_number_of_trials_per_task.Value{1};
    tasks = {'0-back_session', '2-back_session','3-back_session'};
    db_folder = current_protocol.STUDIES;
    NIRS_data_path = fullfile(bst_get('BrainstormUserDir'),'NIRS_Data');

    %////////////////////////////////////////////// NIRS preprocessing
    if (sProcess.options.option_NIRS_preprocessing.Value)
       if (sProcess.options.option_apply_LPF.Value)
        % Process: Low-pass: 0.2Hz
        sFiles = bst_process('CallProcess', 'process_nst_iir_filter', sFiles, [], ...
            'sensortypes',        'NIRS', ...
            'option_filter_type', 1, ...  % bandpass
            'option_keep_mean',   sProcess.options.option_keep_mean.Value, ...
            'option_low_cutoff',  sProcess.options.option_low_cutoff.Value, ...
            'option_high_cutoff', sProcess.options.option_high_cutoff.Value, ...
            'order',              sProcess.options.order.Value, ...
            'overwrite',          0);
        end

        %Import in database
        OutputFiles = modified_import_raw_to_db(sFiles.FileName);

        %Bad trials removal
        if (sProcess.options.bad_trials_removal.Value)
            Detect_bad_trials(db_folder, num_of_trials, tasks, sProcess.options.coefficient_variation.Value{1}, sProcess.options.bad_channels_percentage.Value{1})
        end

        %Generate avg files
        if (sProcess.options.option_NIRS_average.Value)
            Get_individual_avg(db_folder, num_of_trials, tasks)
            %Get_group_avg(db_folder, num_of_participants, num_of_trials, tasks)
        end

        %Export to CSV files
        Export_to_csv(db_folder, num_of_trials, tasks, NIRS_data_path) 
    end
    
    %////////////////////////////////////////////// ML processing
    if (sProcess.options.option_ML_classifier.Value)
        %if (sProcess.options.option_NIRS_preprocessing.Value)
        if (sProcess.options.option_data_type.Value == 'NIRS')
            %Creating All_Data folder for NIRS
            NIRS_All_Data_path = fullfile(NIRS_data_path, 'All_Data');
            if (~exist(NIRS_All_Data_path))
                mkdir(fullfile(NIRS_data_path, 'All_Data'));
            end
            %Call ML script for NIRS
            syscmd = sprintf('python %s %s', 'G:/test_ML.py', bst_get('BrainstormUserDir'));
            [status, commandOut] = system(syscmd);
            commandOut
        end
    end
end

