function GLM_PCA_ANGULAR (regressor_pca_ang,signal)
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
