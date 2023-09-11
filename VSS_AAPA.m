%% code for VSS-WLCAPA
%% coding by Long

clc
clear all
close all

%% parameter setting
N = 8; %filter length
iter = 4000; %iteration
run_num = 100; 
MSE = zeros(1,iter);
MSD = zeros(1,iter);
P = 4; %numbers of projection
mu_vector = zeros(1,iter); % for saving step-size
% AR = [1 -0.95];

for k = 1:run_num
    %%% construct unknown system
    unknown_h1 = randn(N,1) + 1j*randn(N,1);
    unknown_h1 = unknown_h1/norm(unknown_h1);
    unknown_g1 = randn(N,1) + 1j*randn(N,1);
    unknown_g1 = unknown_g1/norm(unknown_g1);
    unknown_w1 = [unknown_h1;unknown_g1];
    
    unknown_h2 =  - unknown_h1;
    unknown_g2 =  - unknown_g1;
    unknown_w2 =  - unknown_w1;
    
    %%% construct complex-valued AR(0.95) input
    x = zeros(1,iter);
    white_input = sqrt(0.75)*randn(1,iter) + sqrt(0.25)*1j*randn(1,iter);
    x(1) = white_input(1);
    %%% generate AR(1)input signal
    for jj = 2:iter
        x(jj) = 0.95*x(jj-1) + white_input(jj);
    end
    
    %%% generate output
    y = [filter(transpose(unknown_h1), 1, x(1:iter/2)) + filter(transpose(unknown_g1), 1, conj(x(1:iter/2))) ...
        filter(transpose(unknown_h2), 1, x(iter/2+1:iter)) + filter(transpose(unknown_g2), 1, conj(x(iter/2+1:iter)))]; %output
    var_noise = 0.01; % variance of noise
    v = sqrt(var_noise/2)*randn(1,iter) + sqrt(var_noise/2)*1j*randn(1,iter);
    d = y + v; % desired signal
    L = length(x);
    
    %%% intialization 
    filterinput = zeros(N,P); %input matrix
    h = zeros(N,1); %initialization of filter
    g = zeros(N,1);
    w = [h;g]; 
    eACLMS = zeros(P,L);
    filteroutput = zeros(P,L);
    estimate_e = 0;
    estimate_a = 0;
    vector = zeros(P,2);
    mid_err = zeros(1,P);
    
    %%% parameters setting
    alfa = 0.95;
    theta = 3;
    
    for ii = P+N:L
        %%% generate input vector
        for m = 1:P
            filterinput(:,m) = transpose(x((ii-m+1):-1:(ii-m+1-N+1)));
        end 
        
        %%% filter output and error
        filteroutput(:,ii) = transpose(filterinput)*h + filterinput'*g;
        eACLMS(:,ii) = transpose(d(ii:-1:ii-P+1)) - filteroutput(:,ii);
        
        %%% estimate for e^2(k)
        estimate_e = alfa*estimate_e + (1-alfa)*norm(eACLMS(:,ii))^2;
        %%% estimate for e_a^2(k)
        t = sqrt(theta*var_noise);
        for kk = 1:P
            vector(kk,:) = [abs(eACLMS(kk,ii))-t 0];  
            mid_err(kk) = sign(eACLMS(kk,ii))*max(vector(kk,:));
        end
        estimate_a = alfa*estimate_a + (1-alfa)*norm(mid_err)^2;
        mu = estimate_a/(estimate_e);
        mu_vector(ii) = mu_vector(ii) + mu; % save step-size
         
        %%%% update of the weight vector
        h = h + mu*conj(filterinput)*(filterinput'*filterinput+transpose(filterinput)*conj(filterinput))^-1*eACLMS(:,ii);
        g = g + mu*filterinput*(filterinput'*filterinput+transpose(filterinput)*conj(filterinput))^-1*eACLMS(:,ii);
        w = [h;g];
        
        %%% MSE and MSD
        MSE(ii) = MSE(ii) + norm(eACLMS(1,ii))^2;
        if ii <= iter/2
           MSD(ii) = MSD(ii) + sum(abs(w - unknown_w1).^2);
        else
           MSD(ii) = MSD(ii) + sum(abs(w - unknown_w2).^2);
        end
    end
end

%%% independent average
MSE = MSE/run_num;
MSD = MSD/run_num;
mu_vector = mu_vector/run_num;

%%% learning curves
figure(1)
plot(10*log10(MSE),'-r');
figure(2)
plot(mu_vector,'c');
figure(3)
plot(10*log10(MSD),'-b');










