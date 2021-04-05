function varargout = process_run_python_ML( varargin )

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


eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    %TOCHECK: how do we limit the input file types (only NIRS data)?
    sProcess.Comment     = 'Train machine learning model';
    sProcess.FileTag     = '';
    sProcess.Category    = 'File';
    sProcess.SubGroup    = 'Capstone';
    sProcess.Index       = 4005; %0: not shown, >0: defines place in the list of processes
    sProcess.Description = 'https://github.com/NK2306/Capstone_490';
    sProcess.isSeparator = 0; 
    
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'import'};
    sProcess.OutputTypes = {'matrix'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 0;
    
    %Definition of the options
    
    %////////////////////////////////////////////////////////// Machine Learning /////////////////////////////////////////////////////
    
    sProcess.options.option_data_type.Comment = 'Type of Data (EEG or NIRS)';
    sProcess.options.option_data_type.Type = 'text';
    sProcess.options.option_data_type.Value = 'EEG';
    
    sProcess.options.option_ML_operation_mode.Comment = 'Operation mode: ';
    sProcess.options.option_ML_operation_mode.Type    = 'combobox';
    sProcess.options.option_ML_operation_mode.Value   = {1, {'Train', 'Test', 'Train-Test split'}};
    
    %To disable when either test or train is selected
    sProcess.options.train_percentage.Comment = 'Percentage of data to be used for training (Train-Test split only): ';
    sProcess.options.train_percentage.Type    = 'value';
    sProcess.options.train_percentage.Value   = {80, '%', 0};
    
    sProcess.options.sliding_windows_method.Comment = 'Use sliding windows method';
    sProcess.options.sliding_windows_method.Type    = 'checkbox';
    sProcess.options.sliding_windows_method.Value   = 0;
    
    sProcess.options.negative_time.Comment = 'Remove negative time';
    sProcess.options.negative_time.Type    = 'checkbox';
    sProcess.options.negative_time.Value   = 0;
    
    sProcess.options.option_sensors_group.Comment = 'Select sensors group: ';
    sProcess.options.option_sensors_group.Type    = 'combobox';
    sProcess.options.option_sensors_group.Value   = {1, {'Frontal', 'Parietal', 'Occupital', 'All'}};
end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end

%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>

    OutputFiles = {};
    current_protocol = bst_get('ProtocolInfo');
    NIRS_data_path = fullfile(bst_get('BrainstormUserDir'),'NIRS_Data');
    
    %////////////////////////////////////////////// ML processing
    %if (sProcess.options.option_NIRS_preprocessing.Value)
    if (sProcess.options.option_data_type.Value == 'NIRS')
        %Creating All_Data folder for NIRS
        NIRS_All_Data_path = fullfile(NIRS_data_path, 'All_Data');
        if (~exist(NIRS_All_Data_path))
            mkdir(fullfile(NIRS_data_path, 'All_Data'));
        end
        %Call ML script for NIRS
        param_x = sProcess.options.option_data_type.Value;
        param_y = sProcess.options.option_sensors_group.Value{1}; %{'Frontal', 'Parietal', 'Occupital', 'All'}
        switch param_y
            case param_y == 1
                param_y = 'frontal';
            case param_y == 2
                param_y = 'parietal';
            case param_y == 3
                param_y = 'occupital';
            otherwise
                param_y = 'all';
        end
        param_z = sProcess.options.sliding_windows_method.Value;
        if (param_z == 0)
            param_z = 'False';
        else
            param_z = 'True';
        end
        param_path = bst_get('BrainstormUserDir');
        param_percentage= sProcess.options.train_percentage.Value{1};
        param_just_do = sProcess.options.option_ML_operation_mode.Value{1}; %{'Train', 'Test', 'Train-Test split'}
        switch param_just_do
            case param_just_do == 1
                param_just_do = 'Train';
            case param_just_do == 2
                param_just_do = 'Test';
            otherwise
                param_just_do = 'tts';
        end


        syscmd = sprintf('python %s %s %s %s %s %d %s', 'G:/Python/ML_models_analysis.py', param_x, param_y, param_z, param_path, param_percentage, param_just_do);
        [status, commandOut] = system(syscmd);
        commandOut
    end
end

