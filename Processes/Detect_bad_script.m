myDataDir = '/NAS/home/kh_guy/Capstone/Brainstorm/brainstorm_db/shin_worload/data';
n_subject = 26;
n_trial = 9;
tasks = {'0-back_session', '2-back_session','3-back_session'};

for i_subject = 1:n_subject
    subject_name =  sprintf('VP%03d-NIRS',i_subject);
    subject_foler= fullfile( myDataDir,subject_name);
    if ~exist(subject_foler)
        %printf('Invalid subject folder');
    return;    
    end
    
    %mkdir(fullfile('/NAS/home/kh_guy/Capstone/CSV_Data', subject_name));
    
    for i_task = tasks
        task_name = char(i_task);
        %save_folder_name = fullfile('/NAS/home/kh_guy/Capstone/CSV_Data', subject_name, task_name);
        %mkdir(save_folder_name);
        
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
                                'coefficient_variation',        10, ...
                                'bad_channels_percentage',      60);
        end
    end
end