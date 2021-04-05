function varargout = process_GLM( varargin )

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


eval(macro_method);
end

%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'GLM';
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
    
    sProcess.options.MC_method.Comment = 'Select motion correction method: ';
    sProcess.options.MC_method.Type    = 'combobox';
    sProcess.options.MC_method.Value   = {1, {'PCA angular', 'Short distance'}};

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
    %Compute(ChanneMat,sData,sProcess.options);
    %RAMY NEEDS TO USE THIS IN HIS CODE
    %[regressor_pca_ang] = process_capst_correlation('Compute', channel,signal, options)
    
    if (sProcess.options.MC_method.Value{1} == 1) %PCA
        pcd_regressor = Compute(ChanneMat,sData,sProcess.options);
        GLM_PCA_ANGULAR (pcd_regressor, sData);
    end
end


%options.windows_length = 50;
%options.option_chan_name = 'S1D1WL685';
%temp = Compute(channel,signal,options);
%GLM_PCA_ANGULAR (temp, signal)
%GLM_SHORT_DISTANCE (signal)

function [regressor_pca_ang] = Compute(channel,signal, options)
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
       [ang_rho(end+1,1) ] = circ_corrcc(baseline(1,:), windowed_angularvalue(1,:));  %X
       [ang_rho(end,2)] = circ_corrcc(baseline(2,:), windowed_angularvalue(2,:));       %Y  
       [ang_rho(end,3)] = circ_corrcc(baseline(3,:), windowed_angularvalue(3,:));       %Z  
      
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

function GLM_PCA_ANGULAR (regressor_pca_ang, signal)
hrf_duration = 30;
fs=10;

time = 0:1/fs:hrf_duration;
% HRF is the sum of two gamma function. 
% Here first gamma peak at 5s; and FWHM is 5.2
% Second gama peak at 15s and is 9s wide.
% We scale the function so it peak at 3\mumol/l
hrf = computeHRF(time, 5, 5.2, 15, 9, 0.1,3);
 
figure; 
plot(time, hrf);
title('hrf');
xlabel('Time(s)'); ylabel('Hb0 (\mu mol/l)')
 
% Generation of a simulated paradigm:
 
t_end       = 105.5; % Duration of the recording: 3 minutes
%t_end=180;
t           = 0:1/fs:t_end;
 
% Generate a paradigm with 15s of task followed by 30 to 60s of task
events      = generateParadigm(t, 15, [30 60],25);
 
% Generate the design matrix and normalize it
Xref= nst_make_event_regressors(events, hrf', t);
 
figure; 
subplot(121)
imagesc(Xref);
ax = gca;
ylabel('time(sample)')
xlabel('regresssors')
colormap('gray')
ax.XTick = [1 2];
ax.XTickLabel = {'Task','Rest'};
subplot(122)
plot(t,Xref);
xlabel('time(s)')
 
% We then normalize 
Xref(:,1)   = Xref(:,1) / norm(Xref(:,1));
Xref(:,2)   = Xref(:,1) / norm(Xref(:,2));
 
% Generate a signal
 
data = zeros(1, length(t));
 
nirs_paradigm =  (Xref*[5;0 ])'; % Signal due to the task
figure;
plot (t,nirs_paradigm)
xlabel('Time(s)');
title('Nirs Paradigm');
% adding noise
noise=signal.F(1,:);
figure;
plot (t,noise)
xlabel('Time(s)');
title('Noise');

SNR_ref   =  10*log10( norm(nirs_paradigm') / norm(noise));
SNR_target= 0; % target SNR in db; 5db is high SNR
 
% Compute a scaling factor to get to the desire SNR
gamma     =  10^( (SNR_ref - SNR_target)/10); 
data(1,:) =  nirs_paradigm + gamma*noise ; 
 
figure; 
plot(t,data)
xlabel('Time(s)');ylabel('Hb0 (\mu mol/l)')
title('Data paradigm+noise*gamma');
% Add paradigm on the plot
hold on; 
axes_lim = ylim(gca);
for i_event = 1:size(events(1).times,2)
    rectangle('Position',[events(1).times(1,i_event) axes_lim(1) events(1).times(2,i_event)-events(1).times(1,i_event) axes_lim(2)-axes_lim(1)],...
        'FaceColor',[1 0 0 0.5], 'EdgeColor','none') 
end

% GLM analysis

% Input:  t: timevector
%         events: time of the task
%         data  : nirs signal
% Output: beta vector
 % Create the design matrix, modelling only the task and normalize it 
X= nst_make_event_regressors(events(1), hrf', t);

pca_angular=regressor_pca_ang(:,1);

X2=[X pca_angular];
%normalize X2
X2 = X2 / norm(X2);
%obtain n and k
[n,k]=size(X2);
%transpose data
y=transpose(data);
%equation for beta
B=inv(transpose(X2)*X2)*transpose(X2)*y
%GLM matlab function
C=glmfit(X2,y)
%residual 
r=y-(X2*B);
%variance of residual
var_residual=(transpose(r)*r)/(n-k);
% variance of paradigm
var_para=inv(transpose(X2)*X2);
%std 
%std= var_residual*var_para
%contrast vector 
column=[1 0];
%transpose contrast vector 
col_t=transpose(column);
%std after adding short distance 
std=var_residual*column*var_para*col_t
sq_std=sqrt(std);
% calculate t value
t_value=(column*B)/sq_std
%calculate p value
p_value= 2*(1-tcdf(abs(t_value),n-1))

%calculating hrf parameters
fitted_value_hrf= X2(:,1)*B(1,:);
figure;
plot(t,fitted_value_hrf)
xlabel('Time(s)')
title("fitted Value hrf" +" "+"t="+t_value+""+"p="+p_value);



column2=[0 1];
col2_t=transpose(column2);
std2=var_residual*column2*var_para*col2_t;
sq_std2=sqrt(std2);
t_value_pca_ang=(column*B)/sq_std2
p_value_pca_ang= 2*(1-tcdf(abs(t_value_pca_ang),n-1))

%Calculating pca ang parameters 
fitted_value_pca_ang= X2(:,2)*B(2,:);
figure;
plot(t,fitted_value_pca_ang)
xlabel('Time(s)')
title("fitted pca ang" +" "+"t="+t_value_pca_ang+""+"p="+p_value_pca_ang);

%Plots
figure;
plot(t,data)
xlabel('Time(s)')
title('data (nirs+noise*gamma)');
figure;
plot(t,noise)
xlabel('Time(s)')
title('noise(signal with motion)')
fitted_value=X2*B;
figure;
plot(t,fitted_value)
xlabel('Time(s)')
title('fitted Value');

data_trans=transpose(data);
residual=data_trans-fitted_value;

figure;
plot(t,residual)
xlabel('Time(s)')
title('Residual');

figure;
tiledlayout(4,1)
nexttile
plot (t,data)
xlabel('Time(s)')
title('Data (nirs*hrf+noise)')
nexttile
plot(t,noise)
xlabel('Time(s)')
title('noise (data with motion)')
nexttile
plot(t,fitted_value)
xlabel('Time(s)')
title('fitted value')
nexttile 
plot(t,residual)
xlabel('Time(s)')
title('residual')

figure;
tiledlayout(2,1)
nexttile
plot (t,data)
xlabel('Time(s)');
title('pca ang+data with motion')
hold on
plot(t,fitted_value,'r')
xlabel('Time(s)');
hold on
plot(t,pca_angular,'y')
xlabel('Time(s)');
legend('data','fitted value','PCA angular'); 
hold off
nexttile 
plot(t,residual,'g')
xlabel('Time(s)');
legend ('residual') 

figure;
tiledlayout(2,1)
nexttile
plot (t,fitted_value)
legend('fitted value')
title('fitted value')
hold on
plot(t,fitted_value_hrf,'r')
legend('fitted value hrf') 
hold on
plot(t,fitted_value_pca_ang,'g')
legend ('fitted pca ang') 
title('fitted values')
hold off
nexttile 
plot(t,residual,'g')
xlabel('Time(s)');
legend ('residual')

end

function GLM_SHORT_DISTANCE (signal)
hrf_duration = 30;
fs=10;

time = 0:1/fs:hrf_duration;
% HRF is the sum of two gamma function. 
% Here first gamma peak at 5s; and FWHM is 5.2
% Second gama peak at 15s and is 9s wide.
% We scale the function so it peak at 3\mumol/l
hrf = computeHRF(time, 5, 5.2, 15, 9, 0.1,3);
 
figure; 
plot(time, hrf);
title('hrf');
xlabel('Time(s)'); ylabel('Hb0 (\mu mol/l)')
 
% Generation of a fake paradigm:
 
t_end       = 105.5; % Duration of the recording: 3 minutes
%t_end=180;
t           = 0:1/fs:t_end;
 
% Generate a paradigm with 15s of task followed by 30 to 60s of task
events      = generateParadigm(t, 15, [30 60],25);
 
% Generate the design matrix and normalize it
Xref= nst_make_event_regressors(events, hrf', t);
 
figure; 
subplot(121)
imagesc(Xref);
ax = gca;
ylabel('time(sample)')
xlabel('regresssors')
colormap('gray')
ax.XTick = [1 2];
ax.XTickLabel = {'Task','Rest'};
subplot(122)
plot(t,Xref);
xlabel('time(s)')
 
% We then normalize 
Xref(:,1)   = Xref(:,1) / norm(Xref(:,1));
Xref(:,2)   = Xref(:,1) / norm(Xref(:,2));
 
% Generate signal
 
data = zeros(1, length(t));
 
nirs_paradigm =  (Xref*[5;0 ])'; % Signal due to the task
figure;
plot (t,nirs_paradigm)
xlabel('Time(s)');
title('Nirs Paradigm');
%adding noise
noise=signal.F(1,:);
figure;
plot (t,noise)
xlabel('Time(s)');
title('Noise');

SNR_ref   =  10*log10( norm(nirs_paradigm') / norm(noise));
SNR_target= 0; % target SNR in db; 5db is high SNR
 
% Compute a scaling factor to get to the desire SNR
gamma     =  10^( (SNR_ref - SNR_target)/10); 
data(1,:) =  nirs_paradigm + gamma*noise ; 
 
figure; 
plot(t,data)
xlabel('Time(s)');ylabel('Hb0 (\mu mol/l)')
title('data');
% Add paradigm on the plot
hold on; 
axes_lim = ylim(gca);
for i_event = 1:size(events(1).times,2)
    rectangle('Position',[events(1).times(1,i_event) axes_lim(1) events(1).times(2,i_event)-events(1).times(1,i_event) axes_lim(2)-axes_lim(1)],...
        'FaceColor',[1 0 0 0.5], 'EdgeColor','none') 
end


% GLM analysis

% Input:  t: timevector
%         events: time of the task
%         data  : nirs signal
% Output: beta vector
 % Create the design matrix, modelling only the task and normalize it 
X= nst_make_event_regressors(events(1), hrf', t);


%create a short distance signal 
shrtdst=transpose(signal.F(13,:));

X2=[X shrtdst];
%normalize X2
X2 = X2 / norm(X2);
% obtain n and k
[n,k]=size(X2);
%transpose data
y=transpose(data);
%equation for beta
B=inv(transpose(X2)*X2)*transpose(X2)*y
%GLM matlab function
C=glmfit(X2,y)
%residual 
r=y-(X2*B);
%variance of residual
var_residual=(transpose(r)*r)/(n-k);
% variance of paradigm
var_para=inv(transpose(X2)*X2);
%contrast vector 
column=[1 0];
%transpose contrast vector 
col_t=transpose(column);
%std after adding short distance 
std=var_residual*column*var_para*col_t
sq_std=sqrt(std);

% calculate t value
t_value=(column*B)/sq_std
%calculate p value
p_value= 2*(1-tcdf(abs(t_value),n-1))

%calculating hrf parameters
fitted_value_hrf= X2(:,1)*B(1,:);
figure;
plot(t,fitted_value_hrf)
xlabel('Time(s)')
title("fitted Value hrf" +" "+"t="+t_value+""+"p="+p_value);

%calculating 
column2=[0 1];
col2_t=transpose(column2);
std2=var_residual*column2*var_para*col2_t;
sq_std2=sqrt(std2);
t_value_shrt=(column*B)/sq_std2
p_value_shrt= 2*(1-tcdf(abs(t_value_shrt),n-1))

%Calculating short distance parameters 
fitted_value_shrt= X2(:,2)*B(2,:);
figure;
plot(t,fitted_value_shrt)
xlabel('Time(s)')
title("fitted Value shrt" +" "+"t="+t_value_shrt+""+"p="+p_value_shrt);

%Plots
figure;
plot(t,data)
xlabel('Time(s)')
title('Signal');
figure;
plot(t,noise)
xlabel('Time(s)')
title('noise (signal with motion)')
fitted_value=X2*B;
figure;
plot(t,fitted_value)
xlabel('Time(s)')
title('fitted Value');

data_trans=transpose(data);
residual=data_trans-fitted_value;

figure;
plot(t,residual)
xlabel('Time(s)')
title('Residual');

figure;
tiledlayout(4,1)
nexttile
plot (t,data)
xlabel('Time(s)')
title('Data (nirs*hrf+noise)')
nexttile
plot(t,noise)
xlabel('Time(s)')
title('noise (data with motion)')
nexttile
plot(t,fitted_value)
xlabel('Time(s)')
title('fitted value')
nexttile 
plot(t,residual)
xlabel('Time(s)')
title('residual')

figure;
tiledlayout(2,1)
nexttile
plot (t,data)
title('shrt dst+data')
hold on
plot(t,fitted_value,'r')
hold on
plot(t,shrtdst,'y')
xlabel('Time(s)');
legend('data','fitted value','short distance'); 
hold off
nexttile 
plot(t,residual,'g')
xlabel('Time(s)');
legend ('residual') 

figure;
tiledlayout(2,1)
nexttile
plot (t,fitted_value)
title('fitted value')
hold on
plot(t,fitted_value_hrf,'r') 
hold on
plot(t,fitted_value_shrt,'g')
title('fitted values')
legend('fitted value', 'fitted value hrf','fitted value shrt dst')
xlabel('Time(s)')
hold off
nexttile 
plot(t,residual,'g')
xlabel('Time(s)');
legend ('residual')

end

function events=generateParadigm(time, task_duration, rest_duration,offset)
% Returned the design matrix
% generate events with the repetition of the following paterns 
% Stim 1 [ 20 - 30s]  - Stim 2 [20-30s] 


    events = db_template('event');
    stim_event_names = {'stim1', 'stim2'};

    events(1).label = stim_event_names{1};
    events(2).label = stim_event_names{2};

    t=offset;
    i_event=1;

    while t + task_duration + rest_duration(2)  < time(end)

        duration_stim1=task_duration;
        duration_stim2= (rest_duration(2)-rest_duration(1)).*rand() + rest_duration(1);

        events(1).times(1,i_event)=t;
        if duration_stim1 > 0
            events(1).times(2,i_event)=t+duration_stim1;
        end
        
        events(2).times(1,i_event)=t+duration_stim1;
        events(2).times(2,i_event)=t+duration_stim1+duration_stim2;

        t=t+duration_stim1+duration_stim2;
        i_event=i_event+1;

    end
    ColorTable = panel_record('GetEventColorTable');
    for i = 1:length(events)
        iColor = mod(i-1, length(ColorTable)) + 1;
        events(i).color = ColorTable(iColor,:);
        events(i).epochs=ones(1, size(events(i).times,2));
   end
    
end

function hrf = computeHRF(t,TTP1, FWHM1, TTP2,FWHM2, gamma,scale)
    % TTP : time to peak
    % FWHM: Full width at half maximum
    % See Deconvolution of hemodynamic responses along the cortical surface
    % using personalized fNIRS by Machodao et al for exemple of value
    % hrf1 = process_nst_simulate_nirs('computeHRF', 5, 5.2, 15, 9, 0.1,3);
    % hrf2 = process_nst_simulate_nirs('computeHRF', 5, 1, 6, 2, 0.1,3);
    % hrf3 = process_nst_simulate_nirs('computeHRF', 5, 10, 0, 0, 0);
    % hrf4 = process_nst_simulate_nirs('computeHRF', 5, 5.2, 15, 9, 0.5,3);
        
    a1  = 8*log(2)*(TTP1^2)/(FWHM1^2);
    b1  = (FWHM1^2) / (8*log(2)*TTP1);
    
    hrf = ((t/TTP1).^a1).*exp(- (t-TTP1)/b1);
    
    if gamma > 0 
        a2  = 8*log(2)*(TTP2^2)/(FWHM2^2);
        b2  = (FWHM2^2) / (8*log(2)*TTP2);
        
        hrf = hrf  - gamma*((t/TTP2).^a2).*exp(- (t-TTP2)/b2);
    end
    
    scale = scale/max(hrf);
    hrf   = scale*hrf;
    
end    
