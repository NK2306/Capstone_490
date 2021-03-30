function Detect_bad_trials(myDataDir, n_trial, tasks, CV_threshold, bad_trial_threshold)

%myDataDir = '/NAS/home/kh_guy/Capstone/Brainstorm/brainstorm_db/shin_worload/data';
%n_subject = 26;
%n_trial = 9;
%tasks = {'0-back_session', '2-back_session','3-back_session'};

    subject_name =  bst_get('Subject').Name;
    subject_foler= fullfile( myDataDir,subject_name);
    if ~exist(subject_foler)
        %printf('Invalid subject folder');
        return;    
    end
    
    for i_task = tasks
        task_name = char(i_task);
        
        for i_trial = 1:n_trial
            trial_name = sprintf('data_%s_trial%03d', task_name, i_trial);
            trial_name_with_extension = strcat(trial_name, ".mat");
            %save_name_with_extension = strcat(trial_name, ".csv");
            data_file = fullfile( subject_foler, task_name, char(trial_name_with_extension));
            %save_file = fullfile( save_folder_name, char(save_name_with_extension));
            fprintf(trial_name);
            %export_data(data_file,[],save_file,'ASCII-CSV-HDR-TR')
            
            sFiles = bst_process('CallProcess', 'process_detectBadTrial', data_file, [], ...
                                'option_coefficient_variation', 1, ...
                                'coefficient_variation',        CV_threshold, ...
                                'bad_channels_percentage',      bad_trial_threshold);
        end
    end
end