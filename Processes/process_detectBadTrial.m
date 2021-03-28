function varargout = process_detectBadTrial( varargin )

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
    sProcess.Comment     = 'Detect bad trial';
    sProcess.FileTag     = '';
    sProcess.Category    = 'File';
    sProcess.SubGroup    = 'Capstone';
    sProcess.Index       = 4000; %0: not shown, >0: defines place in the list of processes
    sProcess.Description = '';
    sProcess.isSeparator = 0; 
    
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data', 'raw'};
    % Definition of the outputs of this process
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    %{
    % Definition of the options
    sProcess.options.text1.Comment   = '<b>Data based detection</b>'; 
    sProcess.options.text1.Type    = 'label';
    
    sProcess.options.option_sci.Comment = 'Channel rejection by Scalp Coupling Index';
    sProcess.options.option_sci.Type    = 'checkbox';
    sProcess.options.option_sci.Value   = 0;
    sProcess.options.option_sci.Controller='sci';
    
    sProcess.options.sci_threshold.Comment = 'SCI threshold:';
    sProcess.options.sci_threshold.Type    = 'value';
    sProcess.options.sci_threshold.Value   = {80, '%', 0};
    sProcess.options.sci_threshold.Class='sci';
    
    sProcess.options.power_threshold.Comment = 'Power threshold:';
    sProcess.options.power_threshold.Type    = 'value';
    sProcess.options.power_threshold.Value   = {10, '%', 0};
    sProcess.options.power_threshold.Class='sci';
    
    sProcess.options.option_remove_saturating.Comment = 'Remove saturating channels';
    sProcess.options.option_remove_saturating.Type    = 'checkbox';
    sProcess.options.option_remove_saturating.Value   = 0;
    sProcess.options.option_remove_saturating.Controller='saturation';
    
    sProcess.options.option_max_sat_prop.Comment = 'Maximum proportion of saturating points';
    sProcess.options.option_max_sat_prop.Type    = 'value';
    sProcess.options.option_max_sat_prop.Value   = {10, '%', 0};
    sProcess.options.option_max_sat_prop.Class   = 'saturation';

    sProcess.options.option_min_sat_prop.Comment = 'Maximum proportion of flooring points';
    sProcess.options.option_min_sat_prop.Type    = 'value';
    sProcess.options.option_min_sat_prop.Value   = {10, '%', 0};
    sProcess.options.option_min_sat_prop.Class   = 'saturation';

    sProcess.options.text2.Comment   = '<b>Montage based detection</b>'; 
    sProcess.options.text2.Type    = 'label';
    
    sProcess.options.option_separation_filtering.Comment = 'Filter channels based on separation';
    sProcess.options.option_separation_filtering.Type    = 'checkbox';
    sProcess.options.option_separation_filtering.Value   = 0;
    sProcess.options.option_separation_filtering.Controller   = 'separation';
    
    sProcess.options.option_separation.Comment = 'Acceptable separation: ';
    sProcess.options.option_separation.Type    = 'range';
    sProcess.options.option_separation.Value   = {[0, 5], 'cm', 2};
    sProcess.options.option_separation.Class   = 'separation';
    
    sProcess.options.text3.Comment   = '<b>Other</b>'; 
    sProcess.options.text3.Type    = 'label';
    
    sProcess.options.auxilary_signal.Comment   = 'Auxilary measurment:';
    sProcess.options.auxilary_signal.Type    = 'combobox';
    sProcess.options.auxilary_signal.Value   = {1, {'Keep all','Remove flat','Remove all'}};
    
    sProcess.options.option_keep_unpaired.Comment = 'Keep unpaired channels';
    sProcess.options.option_keep_unpaired.Type    = 'checkbox';
    sProcess.options.option_keep_unpaired.Value   = 0;
    %}
    
    sProcess.options.option_coefficient_variation.Comment = 'Channel rejection by coefficient variation';
    sProcess.options.option_coefficient_variation.Type    = 'checkbox';
    sProcess.options.option_coefficient_variation.Value   = 0;
    sProcess.options.option_coefficient_variation.Controller='variation';
    
    sProcess.options.coefficient_variation.Comment = 'Channel rejection by coefficient variation';
    sProcess.options.coefficient_variation.Type    = 'value';
    sProcess.options.coefficient_variation.Value   = {15, '%', 0};
    sProcess.options.coefficient_variation.Class='variation';
    
    sProcess.options.bad_channels_percentage.Comment = 'Percentage of bad channels to disqualify a trial:  ';
    sProcess.options.bad_channels_percentage.Type    = 'value';
    sProcess.options.bad_channels_percentage.Value   = {40, '%', 0};
    sProcess.options.bad_channels_percentage.Class='variation';

end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end

%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>
        
    % Load channel file
    ChanneMat = in_bst_channel(sInputs(1).ChannelFile);
    
    % Load recordings
    if strcmp(sInputs.FileType, 'data')     % Imported data structure
        sData = in_bst_data(sInputs(1).FileName);
    elseif strcmp(sInputs.FileType, 'raw')  % Continuous data file       
        sData = in_bst(sInputs(1).FileName, [], 1, 1, 'no');
    end
    %nirs_flags= strcmpi({ChanneMat.Channel.Type}, 'NIRS');
    %nb_channel=sum(nirs_flags);
    
    [isBad] = Compute(sData, ChanneMat, sProcess.options);
    
    
    
    process_detectbad('SetTrialStatus', sInputs.FileName, isBad);
    
    OutputFiles = [];
end


%% ===== Compute =====
function [isBad] = Compute(sData, channel_def, options)
%% Update the given channel flags to indicate which pairs are to be removed:
%% - negative values
%% - saturating
%% - too long or too short separation
%
% Args
%    - nirs_sig: matrix of double, size: time x nb_channels 
%        nirs signals to be filtered
%    - channel_def: struct
%        Defintion of channels as given by brainstorm
%        Used fields: Nirs.Wavelengths, Channel
%    - channels_flags: array of int, size: nb_channels
%        channel flags to update (1 is good, 0 is bad)
%   [- do_remove_neg_channels: boolean], default: 1
%        actually remove pair where at least one channel has negative
%        values
%   [- max_sat_prop: double between 0 and 1], default: 1
%        maximum proportion of saturating values.
%        If 1 then all time series can be saturating -> ignore
%        If 0.2 then if more than 20% of values are equal to the max
%        the pair is discarded.
%   [- min_sat_prop: double between 0 and 1], default: 1
%        maximum proportion of flooring values.
%        If 1 then all time series can be flooring (equal to lowest value) -> ignore
%        If 0.2 then if more than 20% of values are equal to the min value
%        the pair is discarded.
%   [- max_separation_cm: positive double], default: 10
%        maximum optode separation in cm.
%   [- min_separation_cm: positive double], default: 0
%        minimum optode separation in cm.
%   [- invalidate_paired_channels: int, default: 1]
%        When a channel is tagged as bad, also remove the other paired 
%        channels
%   [- nirs_chan_flags: array of bool, default: ones(nb_channels, 1)]
%        Treat only channels where flag is 1. Used to avoid treating
%        auxiliary channels for example.
%  
% Output:
%    - channel_flags: array of int, size: nb_channels
%    - bad_channel_names: cell array of str, size: nb of bad channels

    %prev_channel_flags = sData.ChannelFlag;
    %channel_flags   = sData.ChannelFlag;
    nirs_flags = strcmpi({channel_def.Channel.Type}, 'NIRS');
    
    signal=sData.F';
    nirs_signal=signal(:,nirs_flags);
    
    nb_chnnels=size(signal,2);
    %nb_sample= size(signal,1);
    %nb_nirs  = sum(nirs_flags);
    %neg_channels = nirs_flags & any(signal < 0, 1);
    %channel_flags(neg_channels) = -1;
    %criteria(1,:)= {'negative channels', neg_channels,{} };
    
    if options.option_coefficient_variation.Value
        CV_threshold = options.coefficient_variation.Value{1};
        
        CV = std(nirs_signal,1)./mean(nirs_signal,1).*100;
        
        CV_channels = false(1,nb_chnnels);
        CV_channels(nirs_flags) = CV>CV_threshold ;
         
        percentage = (sum(CV_channels) /nb_chnnels).*100;

        isBad = percentage > options.bad_channels_percentage.Value{1}; %Add this threshold to the GUI
        
        %{
        
        CV_fit = fitdist(CV','Normal');
        channel_flags(CV_channels) = -1;
        
        %create figure
        fig1= figure('visible','off');
        h1_axes=subplot(1,1,1);
        h=histfit(CV',12,'normal');
        h(1).Annotation.LegendInformation.IconDisplayStyle = 'off';
        h(2).Annotation.LegendInformation.IconDisplayStyle = 'off';
        
        line(h1_axes,[CV_threshold, CV_threshold], ylim(gca), 'LineWidth', 2, 'Color', 'b');
        line(h1_axes,[icdf(CV_fit,0.99), icdf(CV_fit,0.99)], ylim(gca), 'LineWidth', 2, 'Color', 'g');
        
        leg=legend(h1_axes,{sprintf('Used threshold: %.1f (Prob of rejection %.3f%%)',CV_threshold,100*cdf(CV_fit,CV_threshold,'upper')), ...
                              sprintf('Suggested threshold: %.3f',icdf(CV_fit,0.99))});
                           
        title(sprintf('CV rejection: %d channels',sum(CV_channels)))        
        criteria(end+1,:)= {'high coeffience variation channels', CV_channels,{h1_axes,leg}};
        
        %}
    end
    
   %removed =  (prev_channel_flags ~= -1 & channel_flags == -1);
   %removed_channel_names = {channel_def.Channel(removed).Name};
   
end
