%% ===== SET STUDY BAD TRIALS =====
% USAGE:  SetTrialStatus(FileNames, isBad)
%         SetTrialStatus(FileName, isBad)
%         SetTrialStatus(BstNodes, isBad)
function output = GetTrialStatus(FileNames)
    % ===== PARSE INPUTS =====
    % CALL: SetTrialStatus(FileName, isBad)
    if ischar(FileNames)
        FileNames = {FileNames};
        [tmp__, iStudies, iDatas] = bst_get('DataFile', FileNames{1});
    % CALL: SetTrialStatus(FileNames, isBad)
    elseif iscell(FileNames)
        % Get studies indices
        iStudies = zeros(size(FileNames));
        iDatas   = zeros(size(FileNames));
        for i = 1:length(FileNames)
            [tmp__, iStudies(i), iDatas(i)] = bst_get('DataFile', FileNames{i});
        end
    % CALL: SetTrialStatus(BstNodes, isBad)
    else
        % Get dependent nodes
        [iStudies, iDatas] = tree_dependencies(FileNames, 'data', [], 1);
        % If an error occurred when looking for the for the files in the database
        if isequal(iStudies, -10)
            bst_error('Error in file selection.', 'Set trial status', 0);
            return;
        end
        % Get study
        sStudies = bst_get('Study', iStudies);
        % Get data filenames
        FileNames = cell(size(iStudies));
        for i = 1:length(iStudies)
            FileNames{i} = sStudies(i).Data(iDatas(i)).FileName;
        end
    end
    
    % Get protocol folders
    ProtocolInfo = bst_get('ProtocolInfo');
    % Get unique list of studies
    uniqueStudies = unique(iStudies);
    % Remove path from all files
    for i = 1:length(FileNames)
        [fPath, fBase, fExt] = bst_fileparts(FileNames{i});
        FileNames{i} = [fBase, fExt];
    end
    
    % ===== CHANGE TRIALS STATUS =====
    % Update each the study
    for i = 1:length(uniqueStudies)
        % === CHANGE STATUS IN DATABASE ===
        % Get files for this study
        iStudy = uniqueStudies(i);
        iFiles = find(iStudy == iStudies);
        % Get study
        sStudy = bst_get('Study', iStudy);
        % Mark trial as bad
        output = [sStudy.Data(iDatas(iFiles)).BadTrial]
end