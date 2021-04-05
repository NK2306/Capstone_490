function Export_to_csv(myDataDir, n_trial, tasks, save_path)

%myDataDir = '/NAS/home/kh_guy/Capstone/Brainstorm/brainstorm_db/shin_worload/data';
%n_subject = 26;
%n_trial = 9;
%tasks = {'0-back_session', '2-back_session','3-back_session'};

    subject_name =  bst_get('Subject').Name;
    subject_folder= fullfile( myDataDir,subject_name);
    if ~exist(subject_folder)
        %printf('Invalid subject folder');
    return;    
    end
    %save_subject_folder = fullfile(bst_get('BrainstormUserDir'),'NIRS-data', subject_name)
    save_subject_folder = fullfile(save_path, subject_name);
    mkdir(save_subject_folder);
    
    for i_task = tasks
        task_name = char(i_task);
        save_folder_name = fullfile(save_subject_folder, task_name);
        mkdir(save_folder_name);
        
        for i_trial = 1:n_trial
            trial_name = sprintf('data_%s_trial%03d', task_name, i_trial);
            trial_name_with_extension = strcat(trial_name, ".mat");
            save_name_with_extension = strcat(trial_name, ".csv");
            data_file = fullfile( subject_folder, task_name, char(trial_name_with_extension));
            save_file = fullfile( save_folder_name, char(save_name_with_extension));
            %printf(trial_name);
            
            if (~GetTrialStatus(data_file))
                export_data(data_file,[],save_file,'ASCII-CSV-HDR-TR');
            end
        end
        
        try
            %try to find avg file
            avg_file_found = dir(fullfile(subject_folder, task_name, '*_average_*.mat'));
            avg_file_found_name = avg_file_found.name;
            [filepath, avg_name, ext] = fileparts(avg_file_found_name);
            
            %preparing the save file name
            avg_name_with_extension = strcat(avg_name, ext);
            avg_save_name_with_extension = strcat(avg_name, ".csv");
            avg_file = fullfile( subject_folder, task_name, char(avg_name_with_extension));
            avg_save_file = fullfile( save_folder_name, char(avg_save_name_with_extension));
            
            %export to CSV
            export_data(avg_file,[],avg_save_file,'ASCII-CSV-HDR-TR')
        catch
            str = strcat("No avg file found");
            warning(str);
        end
    end
end