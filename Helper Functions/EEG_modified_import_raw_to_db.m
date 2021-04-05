function NewFiles = modified_import_raw_to_db( DataFile, strNBack, badtrial_check, BTlow, BThigh, average_check)
% IMPORT_RAW_TO_DB: Import in the database some blocks of recordings from a continuous file already linked to the database.
%
% USAGE:  NewFiles = import_raw_to_db( DataFile )

% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2020 University of Southern California & McGill University
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
% Authors: Francois Tadel, 2011-2019


% ===== GET FILE INFO =====
% Get study description
[sStudy, iStudy, iData] = bst_get('DataFile', DataFile);
if isempty(sStudy)
    error('File is not registered in the database.');
end
% Is it a "link to raw file" or not
isRaw = strcmpi(sStudy.Data(iData).DataType, 'bandpassed_eeg');
% Get subject index
[sSubject, iSubject] = bst_get('Subject', sStudy.BrainStormSubject);
% Progress bar
bst_progress('start', 'Import raw file', 'Processing file header...');
% Read file descriptor
DataMat = in_bst_data(DataFile);
% Read channel file
ChannelFile = bst_get('ChannelFileForStudy', DataFile);
ChannelMat = in_bst_channel(ChannelFile);
% Get sFile structure
if isRaw
    sFile = DataMat.F;
else
    sFile = in_fopen(DataFile, 'BST-DATA');
end

%ImportOptions
My_imported_options = db_template('ImportOptions');
My_imported_options.ImportMode = 'Event'; %split using events

if (strNBack == '0back')
    setnback_trials =strcmp({sFile.events.label}, 'S 16');
    My_imported_options.events = sFile.events(setnback_trials);
elseif (strNBack == '2back')
    setnback_trials =strcmp({sFile.events.label}, 'S 48');
    My_imported_options.events = sFile.events(setnback_trials);
elseif (strNBack == '3back')
    setnback_trials =strcmp({sFile.events.label}, 'S 80');
    My_imported_options.events = sFile.events(setnback_trials);
end
My_imported_options.TimeRange = sFile.prop.times;
My_imported_options.EventsTimeRange = [-0.1, 1];
My_imported_options.SplitRaw = 0;
My_imported_options.SplitLength = [];
My_imported_options.Resample = 1;
My_imported_options.ResampleFreq = [200];
My_imported_options.RemoveBaseline = 'time'; %Baseline correction option
My_imported_options.BaselineRange = [-0.1,0];
My_imported_options.CreateConditions = 1;
My_imported_options.DisplayMessages = 0;

% Import file

NewFiles = import_data(sFile, ChannelMat, sFile.format, [], iSubject, My_imported_options, sStudy.DateOfStudy);

         if (badtrial_check)
             %INSERT TRIAL REJECTION
             NewFiles = bst_process('CallProcess', 'process_detectbad', sFiles, [], ...
             'timewindow', [-0.1, 1], ...
             'meggrad',    [0, 0], ...
             'megmag',     [0, 0], ...
             'eeg',        [BTLow,BThigh], ...
             'ieeg',       [0, 0], ...
             'eog',        [0, 0], ...
             'ecg',        [0, 0], ...
             'rejectmode', 2);  % Reject the trial fully if bad (not only the trials)

                % Save and display report
                %ReportFile = bst_report('Save', sFiles);
                %bst_report('Open', ReportFile);
         end
         
         if (average_check)
             %INSERT TRIALS AVERAGE
             NewFiles = bst_process('CallProcess', 'process_average', NewFiles, [], ...
            'avgtype',       5, ...  % By trial group (folder average)
            'avg_func',      1, ...  % Arithmetic average:  mean(x)
            'weighted',      0, ...
            'keepevents',    0);
         
            % Save and display report
            %ReportFile = bst_report('Save', sFiles);
            % bst_report('Open', ReportFile);
            % bst_report('Export', ReportFile, ExportDir);
         end

    if (strNBack == '2back')
        setnback_trials =strcmp({sFile.events.label}, 'S 64');
        My_imported_options.events = sFile.events(setnback_trials);
        NewFiles = import_data(sFile, ChannelMat, sFile.format, [], iSubject, My_imported_options, sStudy.DateOfStudy);
    elseif (strNBack == '3back')
        setnback_trials =strcmp({sFile.events.label}, 'S 96');
        My_imported_options.events = sFile.events(setnback_trials);
        NewFiles = import_data(sFile, ChannelMat, sFile.format, [], iSubject, My_imported_options, sStudy.DateOfStudy);
    end

         if (badtrial_check)
             %INSERT TRIAL REJECTION
             NewFiles = bst_process('CallProcess', 'process_detectbad', NewFiles, [], ...
             'timewindow', [-0.1, 1], ...
             'meggrad',    [0, 0], ...
             'megmag',     [0, 0], ...
             'eeg',        [BTLow,BThigh], ...
             'ieeg',       [0, 0], ...
             'eog',        [0, 0], ...
             'ecg',        [0, 0], ...
             'rejectmode', 2);  % Reject the trial fully if bad (not only the trials)

                % Save and display report
                %ReportFile = bst_report('Save', sFiles);
                %bst_report('Open', ReportFile);
         end
         
         if (average_check)
             %INSERT TRIALS AVERAGE
             NewFiles = bst_process('CallProcess', 'process_average', NewFiles, [], ...
            'avgtype',       5, ...  % By trial group (folder average)
            'avg_func',      1, ...  % Arithmetic average:  mean(x)
            'weighted',      0, ...
            'keepevents',    0);
         
            % Save and display report
            %ReportFile = bst_report('Save', sFiles);
            % bst_report('Open', ReportFile);
            % bst_report('Export', ReportFile, ExportDir);
         end



        
         
         
                  
% If only one file imported: Copy linked videos in destination folder
if (length(NewFiles) == 1) && ~isempty(sStudy.Image)
    process_import_data_event('CopyVideoLinks', NewFiles{1}, sStudy);
end

%ReportFiles = bst_report('Save', NewFiles); %SAVING 60 FILES

% Save database
db_save();