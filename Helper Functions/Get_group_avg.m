function Get_group_avg(myDataDir, n_subject, n_trial, tasks)

%myDataDir = '/NAS/home/kh_guy/Capstone/Brainstorm/brainstorm_db/shin_worload/data';
%n_subject = 26;
%n_trial = 9;
%tasks = {'0-back_session', '2-back_session','3-back_session'};

for i_task = tasks
        sFiles = {};
        task_name = char(i_task);
        
    for i_subject = 1:n_subject
        subject_name =  sprintf('VP%03d-NIRS',i_subject);
        subject_foler= fullfile( myDataDir,subject_name);
        
        
        for i_trial = 1:n_trial
            trial_name = sprintf('data_%s_trial%03d', task_name, i_trial);
            trial_name_with_extension = strcat(trial_name, ".mat");
            data_file = fullfile( subject_foler, task_name, char(trial_name_with_extension));
            fprintf(trial_name);
            
            if (~GetTrialStatus(data_file))
                sFiles{end+1} = data_file;
            end
        end               
    end
    
    sFiles = bst_process(   'CallProcess', 'process_average', sFiles, [], ...
                                'avgtype',       1, ...  % Everything
                                'avg_func',      6, ...  % Arithmetic average + Standard deviation
                                'weighted',      0, ...
                                'keepevents',    0);
end