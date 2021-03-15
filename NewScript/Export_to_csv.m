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
    
    mkdir(fullfile('/NAS/home/kh_guy/Capstone/CSV_Data', subject_name));
    
    for i_task = tasks
        task_name = char(i_task);
        save_folder_name = fullfile('/NAS/home/kh_guy/Capstone/CSV_Data', subject_name, task_name);
        mkdir(save_folder_name);
        
        for i_trial = 1:n_trial
            trial_name = sprintf('data_%s_trial%03d', task_name, i_trial);
            trial_name_with_extension = strcat(trial_name, ".mat");
            save_name_with_extension = strcat(trial_name, ".csv");
            data_file = fullfile( subject_foler, task_name, char(trial_name_with_extension));
            save_file = fullfile( save_folder_name, char(save_name_with_extension));
            %printf(trial_name);
            if (GetTrialStatus(data_file) == 0)
                export_data(data_file,[],save_file,'ASCII-CSV-HDR-TR')
            end
        end
        
        avg_name = sprintf('data_%s_average_210315_1325', task_name);
        avg_name_with_extension = strcat(avg_name, ".mat");
        avg_save_name_with_extension = strcat(avg_name, ".csv");
        avg_file = fullfile( subject_foler, task_name, char(avg_name_with_extension));
        avg_save_file = fullfile( save_folder_name, char(avg_save_name_with_extension));
        
        try
            export_data(avg_file,[],avg_save_file,'ASCII-CSV-HDR-TR')
        catch
            str = strcat("No avg file named: ", avg_name);
            warning(str);
        end
        
    end
end