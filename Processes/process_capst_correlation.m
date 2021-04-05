function varargout = process_capst_correlation( varargin )

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
% Authors: Edouard Delaire, 2020;
%%Mdified by Giselt


eval(macro_method);
end
%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'Circular Correlation';
    sProcess.FileTag     = '';
    sProcess.Category    = 'File';
    sProcess.SubGroup    = 'Capstone';
    sProcess.Index       = 4004; %0: not shown, >0: defines place in the list of processes
    sProcess.Description = '';
    sProcess.isSeparator = 0;  
    
     
    % Add option to the process
    sProcess.options.option_chan_name.Comment = 'Channel event name: ';
    sProcess.options.option_chan_name.Type    = 'text';
    sProcess.options.option_chan_name.Value   = 'S1D1WL685';
    
    
    sProcess.options.windows_length.Comment = 'Windows length';
    sProcess.options.windows_length.Type    = 'value';
    sProcess.options.windows_length.Value   = {10,'s',0};

    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data', 'raw'};
    % Definition of the outputs of this process
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end

%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>
    OutputFiles = {};
    
    % Load channel file
    ChanneMat = in_bst_channel(sInputs(1).ChannelFile);
    
    % Load recordings
    if strcmp(sInputs.FileType, 'data')     % Imported data structure
        sData = in_bst_data(sInputs(1).FileName);
    elseif strcmp(sInputs.FileType, 'raw')  % Continuous data file       
        sData = in_bst(sInputs(1).FileName, [], 1, 1, 'no');
    end
    % Assume 10Hz
    %RAMY NEEDS TO USE THIS THE FOLLOWING TWO LINES IN HIS CODE
    sProcess.options.windows_length=sProcess.options.windows_length.Value{1}*10;
    sProcess.options.option_chan_name=sProcess.options.option_chan_name.Value;
    Compute(ChanneMat,sData,sProcess.options);
    %RAMY NEEDS TO USE THIS IN HIS CODE
    %[regressor_pca_ang] = process_capst_correlation('Compute', channel,signal, options)
end
%% ===== Compute =====
function [regressor_pca_ang regressor_pca_corr] = Compute(channel,signal, options)
    windows_length = options.windows_length;   
    idx= contains({channel.Channel.Name},options.option_chan_name);        
    n_sample = length(signal.Time);
    n_channel = size(signal.F,1);
    mov_mean= zeros(n_sample,n_channel); 
    ang_x_index = contains({channel.Channel.Name},'angular  x');
    ang_x_index = find(ang_x_index == 1);
    radSig = circ_ang2rad(signal.F(ang_x_index:ang_x_index+2,:));
    baseline=radSig(:,2:windows_length+2);

    distance = [];
    ang_rho = [];
    nirs_rho = [];
    p_value = [];

    for i = 1:n_sample
       windows_start =  max( 1,i - windows_length/2);
       windows_end   =  min( length(signal.Time), i + windows_length/2);
       nirs_windows  = signal.F(idx,windows_start:windows_end);

       if (windows_end<=windows_length)
           windowed_angularvalue = [zeros(3,windows_length-windows_end+1),radSig(:,windows_start:windows_end)];
       elseif (windows_end == n_sample)
           windowed_angularvalue = [radSig(:,windows_start:windows_end),zeros(3,windows_length-windows_end+windows_start)];
       else
          windowed_angularvalue = radSig(:,windows_start:windows_end);
       end    
       
       %Circ correlation of XX,YY & ZZ
       [ang_rho(end+1,1),p_value(end+1,1) ] = circ_corrcc(baseline(1,:), windowed_angularvalue(1,:));  %X
       [ang_rho(end,2),p_value(end,2)] = circ_corrcc(baseline(2,:), windowed_angularvalue(2,:));       %Y  
       [ang_rho(end,3),p_value(end,3)] = circ_corrcc(baseline(3,:), windowed_angularvalue(3,:));       %Z  
      
       mov_mean(i,:) = mean(nirs_windows);
    end
    ang_rho (isnan(ang_rho) )=0;
    [regressor_pca_corr,~,~] = svd((ang_rho),'ECON');
    [regressor_pca_ang,~,~] = svd((signal.F(ang_x_index:ang_x_index+2,:))','ECON');

%% ===== Plot =====
    %Plotting the raw data nirs and angular (x,y,z)
    figure; hold on; grid on;
    subplot(3,1,1)
    hold on;
    plot(signal.Time,normalized(signal.F(idx,:))+2,'LineWidth',1.5) %%NIRS SIGNAL 
    plot(signal.Time,normalized(signal.F(ang_x_index,:))+1,'LineWidth',1.5)
    plot(signal.Time,normalized(signal.F(ang_x_index+1,:))+1,'LineWidth',1.5)
    plot(signal.Time,normalized(signal.F(ang_x_index+2,:))+1,'LineWidth',1.5)
    title(strcat(options.option_chan_name, ' :Channel used', '- Graph showing the raw data (Angular) along time'))
    legend('nirs','raw x','raw y', 'raw z')
    xlabel('Time(s)')

    %Plotting the circular correlation between the baseline and the windowedvalue for(x,y,z)
    subplot(3,1,2)
    hold on;
    plot(signal.Time,(ang_rho(:,1)),'LineWidth',1.5)  %%XX
    plot(signal.Time, (ang_rho(:,2)),'LineWidth',1.5) %%YY
    plot(signal.Time,(ang_rho(:,3)),'LineWidth',1.5)  %%ZZ
    title(strcat(options.option_chan_name, ':Channel used', '- Graph showing the Circular correlation between the baseline and the windowedvalue of XX, YY and ZZ along time '))
    legend('circ corr xx', 'circ corr yy', 'circ corr zz')
    xlabel('Time(s)')
    
    %Plotting the PCA of the correlations(XX,YY,ZZ) and the angular values(angular z, angular y, angular z)
    subplot(3,1,3)
    hold on;
    plot(signal.Time,(regressor_pca_corr(:,1))-1,'LineWidth',1.5)
    plot(signal.Time,(regressor_pca_ang(:,1))-1,'LineWidth',1.5)
    title(strcat(options.option_chan_name, ' :Channel used', '- Graph showing the PCA of the correlation and the PCA of the angular values along time '))
    legend('PCA of Correlation', 'PCA or raw angular values')
    xlabel('Time(s)')

end
function result = normalized(original)
    my_min=min(original);
    my_max=max(original);
    range=my_max-my_min;
    offset=my_min+range/2;
    result=original-offset;
    result=result/range;
end
