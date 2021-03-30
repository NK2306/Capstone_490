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
    sProcess.Comment     = 'NIRS preprocess for Capstone490';
    sProcess.FileTag     = '';
    sProcess.Category    = 'File';
    sProcess.SubGroup    = 'Capstone';
    sProcess.Index       = 4000; %0: not shown, >0: defines place in the list of processes
    sProcess.Description = 'NIRS preprocessing for Capstone 490 project';
    sProcess.isSeparator = 0; 
    
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data', 'raw'};
    % Definition of the outputs of this process
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    %Definition of the options
    sProcess.options.option_keep_mean.Comment = 'Keep mean';
    sProcess.options.option_keep_mean.Type    = 'checkbox';
    sProcess.options.option_keep_mean.Value   = 1;
    
    sProcess.options.option_low_cutoff.Comment = 'Lower cutoff frequency (0=disable):';
    sProcess.options.option_low_cutoff.Type    = 'value';
    sProcess.options.option_low_cutoff.Value   = {0.01, '', 4};
    
    sProcess.options.option_high_cutoff.Comment = 'Upper cutoff frequency (0=disable):';
    sProcess.options.option_high_cutoff.Type    = 'value';
    sProcess.options.option_high_cutoff.Value   = {0.5, '', 4}; 
    
    sProcess.options.order.Comment = 'Filter order:';
    sProcess.options.order.Type    = 'value';
    sProcess.options.order.Value   = {3, '', 0};
    
    % === Display properties
    sProcess.options.display.Comment = {'process_nst_iir_filter(''DisplaySpec'',iProcess,sfreq);', '<BR>', 'View filter response'};
    sProcess.options.display.Type    = 'button';
    sProcess.options.display.Value   = [];
end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end

%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>

        %Low-pass filter
        sFiles = sInputs.FileName;
        % Start a new report
        %bst_report('Start', sFiles);

        % Process: Low-pass: 0.2Hz
        sFiles = bst_process('CallProcess', 'process_nst_iir_filter', sFiles, [], ...
            'sensortypes',        'NIRS', ...
            'option_filter_type', 1, ...  % bandpass
            'option_keep_mean',   sProcess.options.option_keep_mean.Value, ...
            'option_low_cutoff',  sProcess.options.option_low_cutoff.Value, ...
            'option_high_cutoff', sProcess.options.option_high_cutoff.Value, ...
            'order',              sProcess.options.order.Value, ...
            'overwrite',          0);

        % Save and display report
        %ReportFile = bst_report('Save', sFiles);
        %bst_report('Open', ReportFile);
        
        %Import in database
        OutputFiles = modified_import_raw_to_db(sFiles.FileName);
end

