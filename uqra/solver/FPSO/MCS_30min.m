
clear;clc
addpath('./Data')
load seeds;
format short
rng('default');

global Diag_surge N N_LF Snn TFv dw dw_LF w w_LF RAO wmin wmax Hx

load freq_data
FD_FPSO_LF


% ---
% QTF
% ---

Hx=(-w_LF.^2*M(1,1)+i*w_LF*B(1,1)+K(1,1)).^-1;
Hx(N)=0;

%% EC samples
% load Kvitebjorn_EC_T50
% n_short_term = 20;
% U = data(1:2,:);
% X = data(3:4,:);
% n_samples = size(U,2);
% 
% extremes= zeros(n_short_term, n_samples);
% 
% for r = 1: n_short_term
%     
%     for i=1:n_samples
%         u1 = U(1,i);
%         u2 = U(2,i);
%         Hs = X(1,i);
%         Tp = X(2,i);
%         rng(seeds(r));
%         xinput(1,1:2)=[Hs, Tp];
%         xinput(1,3:1922)=normrnd(0,1,1,960*2);
%         extremes(r,i)=Glimitmax(xinput); 
% 
%     end
%     if mod(i,n_samples*0.01) == 0
%         fprintf('%i  / %i-> %0.2f%% \n', i, n_samples, i/n_samples*100)
%     end
% end
% save('Kvitebjorn_EC_T50.mat','extremes', 'U', 'X')

%% MCS samples
% n_short_term = 1;
% n_samples = 2000000;
% fprintf('Generating Samples...')
% rng('shuffle')
% U = normrnd(0,1,2,n_samples);
% X = zeros(2, n_samples);
% for i=1:n_samples
%     u1 = U(1,i);
%     u2 = U(2,i);
%     Hs = incdfHs(u1);
%     Tp = incdfTp(u2,u1);
%     X(:,i) = [Hs,Tp];
% end
% fprintf('Running Solver...')
% extremes= zeros(n_short_term, n_samples);
% for r = 1: n_short_term
%     
%     for i=1:n_samples
%         u1 = U(1,i);
%         u2 = U(2,i);
%         Hs = X(1,i);
%         Tp = X(2,i);
%         xinput(1,1:2)=[Hs, Tp];
%         xinput(1,3:1922)=normrnd(0,1,1,960*2);
%         extremes(r,i)=Glimitmax(xinput); 
% 
%     end
%     if mod(i,n_samples*0.01) == 0
%         fprintf('%i  / %i-> %0.2f%% \n', i, n_samples, i/n_samples*100)
%     end
% end
% save('DoE_Mcs2E6R1.mat','extremes', 'U', 'X')

%% Test data
% load DoE_McsE7R0
%  
% u = data(1:2,:);
% x = 0*u;
% n_short_term = 1;
% n_samples = 1000000;
% fprintf('Mapping Samples to X space ...\n')
% 
% for i=1:size(u,2)
%     u1 = u(1,i);
%     u2 = u(2,i);
%     Hs = incdfHs(u1);
%     Tp = incdfTp(u2,u1);
%     x(:,i) = [Hs,Tp];
% end
% 
% fprintf('Running Solver\n')
% y = zeros(n_short_term, n_samples);
% 
% for r = 1: n_short_term
%     
%     for i=n_samples:2*n_samples
%         
%         u1 = u(1,i);
%         u2 = u(2,i);
%         Hs = x(1,i);
%         Tp = x(2,i);
%         rng(rng(seeds(r)));
%         xinput(1,1:2)=[Hs, Tp];
%         xinput(1,3:1922)=normrnd(0,1,1,960*2);
%         y(r,i)=Glimitmax(xinput); 
% 
%         if mod(i,n_samples*0.001) == 0
%             fprintf('%i  / %i-> %0.2f%% \n', i, n_samples, i/n_samples*100)
%         end
%     end
% end
% save('DoE_McsE7R0_y2.mat','y', 'u', 'x')

%% %%%% DoE samples
n_short_term = 20;
optimality   = 'D';
ndim         = 2;
% oversampling_alpha = 2;
poly_orders  = 2:11;
data_cand    = load('DoE_McsE7R0.mat');
data_cand    = data_cand.data;
oed_data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/OED';
y50_EC = [15.64597786  3.56292327 -3.1060649  11.26686136 11.4440152];

for p  = poly_orders(1):poly_orders(end)
    
    formatSpec = 'DoE_McsE7R0_2Heme%d_%s.mat';
    filename   = sprintf(formatSpec, p, optimality);
    filein     = fullfile(oed_data_dir, filename);
    doe_data   = load(filein);
    doe_idx    = doe_data.data;
    fprintf('>>> loading %s \n', filename)
    P = factorial(p+ndim)/factorial(p)/factorial(ndim);
%     nsamples = round(P * oversampling_alpha);
%     u_train  = data_cand(1:2,doe_idx(1:nsamples));
    u_train  = data_cand(1:2,doe_idx);
    u_train  = [u_train, u_train + y50_EC(2:3)'];
    x_train  = 0*u_train;
    
    for i=1:size(u_train,2)
        u1 = u_train(1,i);
        u2 = u_train(2,i);
        Hs = incdfHs(u1);
        Tp = incdfTp(u2,u1);
        x_train(:,i) = [Hs,Tp];
    end

    fprintf('Running Solver\n')
    
    y= zeros(n_short_term, size(u_train, 2));
    for r = 1: n_short_term
        
        for i=1:size(u_train, 2)
            u1 = u_train(1,i);
            u2 = u_train(2,i);
            Hs = x_train(1,i);
            Tp = x_train(2,i);
            rng(seeds(r));
            xinput(1,1:2)=[Hs, Tp];
            xinput(1,3:1922)=normrnd(0,1,1,960*2);
            y(r,i)=Glimitmax(xinput); 

            if mod(i,size(u_train, 2)*0.01) == 0
                fprintf('   - %i  / %i-> %0.2f%% \n', i, size(u_train, 2), i/size(u_train, 2)*100)
            end
        end
    end

    formatSpec = 'DoE_McsE5R0_2Heme%d_%s_y.mat';
    filename   = sprintf(formatSpec, p, optimality);
%     filein     = fullfile(oed_data_dir, filename);
    u = u_train;
    x = x_train;
    save(filename,'y', 'u', 'x')    

end

U = data;
X = 0*U;
n_short_term = 1;
n_samples = size(U,2);
fprintf('Mapping Samples to X space ...\n')
% 




