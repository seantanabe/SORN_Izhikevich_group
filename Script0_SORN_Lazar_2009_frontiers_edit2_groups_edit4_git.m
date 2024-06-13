% figure 4 occluder task
clear all

pred_future = 1

N_E = 400;
N_I = 0.2*N_E;
N_U = 15;
T_E_max = 0.75;
T_I_max = 1.4;
lambda_W = 10;
n_STDP = 0.001;
n_IP = 0.001;
H_IP = 2*(N_U/N_E);

% row to, col from
W_EE = zeros(N_E,N_E);
tmp = find(~eye(N_E));
W_EE(tmp(randperm(length(tmp),N_E*lambda_W))) = 1;
W_EE = W_EE.*unifrnd(0,1,N_E,N_E);
tmp = sum(W_EE,2);
for i = 1:N_E
    W_EE(i,:) = W_EE(i,:)/tmp(i);
end
ind_w = find(W_EE>0);

W_EI = unifrnd(0,1,N_E,N_I);
tmp = sum(W_EI,2);
for i = 1:N_E
    W_EI(i,:) = W_EI(i,:)/tmp(i);
end

W_IE = unifrnd(0,1,N_I,N_E);
tmp = sum(W_IE,2);
for i = 1:N_I
    W_IE(i,:) = W_IE(i,:)/tmp(i);
end

T_E = unifrnd(0,T_E_max,N_E,1);
T_I = unifrnd(0,T_I_max,N_I,1);

v_U = zeros(N_E,9);
for i = 1:9
    v_U(randperm(N_E,N_U),i) = 1;
end

% words = [1 2 3 4 5 6 7 8;
%     8 7 6 5 4 3 2 1;
%     1 9 9 9 9 9 9 8;
%     8 9 9 9 9 9 9 1];

words = [1 2 3 4 5 6 7 8;
    8 7 6 5 4 3 2 1;
    4 5 3 6 2 1 8 7;
    7 8 1 2 6 3 5 4];

%% SORN

run_t = 50000;
U = [];
for w = randi([1 size(words,1)],1,run_t/size(words,2));
    U = [U words(w,:)];
end
x_t = zeros(N_E,run_t+1);
x = zeros(N_E,1); y = zeros(N_I,1);
x_t(:,1) = x;
for t = 1:run_t
    disp(['SORN step ' num2str(t)])
    x = (W_EE*x-W_EI*y+v_U(:,U(t))-T_E) > 0;
    y = (W_IE*x-T_I) > 0;
    
    x_t(:,t+1) = x;
    
    %STDP
    if t > 1
        delta_W_EE = n_STDP*(x_t(:,t)*x_t(:,t-1)' - x_t(:,t-1)*x_t(:,t)');
        W_EE(ind_w) = W_EE(ind_w) + delta_W_EE(ind_w);
        
        W_EE   = min(W_EE, ones(N_E,N_E));
        W_EE   = max(W_EE, zeros(N_E,N_E));
    end
    
    %SN
    W_EE = spdiags(sum(W_EE,2), 0, N_E, N_E) \ W_EE ;
    
    %IP
    T_E = T_E + n_IP*(x-H_IP);
    T_E   = max(T_E, zeros(size(T_E)));
end

%
% figure
% imagesc(x_t)

%% train readout

run_t = 50000;
U = [];
for w = randi([1 size(words,1)],1,run_t/size(words,2))
    U = [U words(w,:)];
end
x_prime_t = zeros(N_E,run_t+1);
x_prime = zeros(N_E,1);
x_prime_t(:,1) = x_prime;
for t = 1:run_t
    disp(['train step ' num2str(t)])
    %     x = (W_EE*x-W_EI*y+v_U(:,U(t))-T_E) > 0;
    x = (W_EE*x-W_EI*y+(rand(N_E,1) < 0.1)-T_E) > 0;
    y = (W_IE*x-T_I) > 0;
    
    x_prime = (W_EE*x-W_EI*y-T_E) > 0;
    x_prime_t(:,t+1) = x_prime;
end

% figure
% imagesc(x_prime_t)

% x_prime_t = [ones(1,size(x_prime_t,2)); x_prime_t];
% ind = setdiff(1:run_t, 1:size(words,2):run_t);
% readout = pinv(x_prime_t(:,ind+pred_future)')*U(ind)';

%% find group

% figure
% histogram(nonzeros(W_EE(:)))

strong_W_EE = prctile(nonzeros(W_EE(:)),95);
anchors = find(sum(strong_W_EE < W_EE,1) > 1);
groups_step = cell(length(anchors),3);
groups_step_conn = cell(length(anchors),3);
for an = anchors
    an_n = find(anchors == an);
    
    step1 = find(W_EE(:,an));
    step2 = find(sum(W_EE(:,step1) > 0,2) > 1);
    group_step2 = step2(find(sum(W_EE(step2,step1),2)-T_E(step2) > 0));
    group_step1 = step1(find(sum(W_EE(group_step2,step1),1) > 0));
    
    % specific connections
    [I_tmp,J_tmp] = find(W_EE(group_step1,an) > 0);
    if size(I_tmp,1) < size(I_tmp,2)
        I_tmp = I_tmp'; J_tmp = J_tmp';
    end
    groups_step_conn{an_n,1} = [an(J_tmp) group_step1(I_tmp)];
    [I_tmp,J_tmp] = find(W_EE(group_step2,group_step1) > 0);
    if size(I_tmp,1) < size(I_tmp,2)
        I_tmp = I_tmp'; J_tmp = J_tmp';
    end
    groups_step_conn{an_n,2} =[group_step1(J_tmp) group_step2(I_tmp)];
    
    groups_step{an_n,1} = an;
    groups_step{an_n,2} = group_step1;
    groups_step{an_n,3} = group_step2;
    
    count = 0;
    group_stepi1 = group_step2;
    stepi1 = step2;
    while ~isempty(group_stepi1)
        if count > 15 % N_E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% not sure what this value should be
            break
        end
        count = count + 1;
        
        stepi2 = find(sum(W_EE(:,group_stepi1) > 0,2) > 0);
        group_stepi2 = stepi2(find(sum(W_EE(stepi2,stepi1),2)-T_E(stepi2) > 0));
        
        %specific connections
        [I_tmp,J_tmp] = find(W_EE(group_stepi2,group_stepi1) > 0);
        if size(I_tmp,1) < size(I_tmp,2)
            I_tmp = I_tmp'; J_tmp = J_tmp';
        end
        groups_step_conn{an_n,2+count} = [group_stepi1(J_tmp) group_stepi2(I_tmp)];
        
        groups_step{an_n,3+count} = group_stepi2;
        
        group_stepi1 = group_stepi2;
        stepi1 = stepi2;
    end
end

% group property
groups = cell(length(anchors),2);
for row = 1:size(groups_step,1)
    tmp = [];
    for col = 1:size(groups_step,2)
        tmp = [tmp; groups_step{row,col}];
    end
    groups{row,1} = unique(tmp);
    groups{row,2} = length(unique(tmp));
end

% group index in activity
slack = 3
for row = 1:size(groups_step,1)
    kernel = zeros(N_E,size(groups_step,2)-1);
    for st = 1:(size(groups_step,2)-1)
        kernel(groups_step{row,st},st) = 1;
    end
    
    polych = NaN(1,size(x_prime_t,2)-size(kernel,2)+1);
    for con = 1:(size(x_prime_t,2)-size(kernel,2)+1)
        polych(1,con) = sum(sum(x_prime_t(:,con:(con+size(kernel,2)-1)).*kernel));
    end
    
    poly_ind = find(polych >= (sum(sum(kernel))-slack));
    groups{row,3} = length(poly_ind);
end
tmp = find([groups{:,2}] > 1)
[groups{tmp,3}]

%% plot group in activity 

disp_row = [15 12 22];
% disp_row = [19 7 14];
disp_N_ch = 100;

% choose channels to display, shuffle
disp_ch = [];
for di = disp_row
    disp_ch = [disp_ch; groups{di,1}];
end
disp_ch = unique(disp_ch);
other_ch = setdiff(1:N_E, disp_ch);
other_ch = other_ch(randperm(length(other_ch)));
tmp_N = disp_N_ch - length(disp_ch);
disp_ch_all = [disp_ch; other_ch(1:tmp_N)'];
disp_ch_all = disp_ch_all(randperm(length(disp_ch_all)));

J_ker_poly_disp = []; I_ker_poly_disp = [];
for di = 1:length(disp_row)
    kernel = zeros(N_E,size(groups_step,2)-1);
    for st = 1:(size(groups_step,2)-1)
        kernel(groups_step{disp_row(di),st},st) = 1;
    end
    
    polych = NaN(1,size(x_prime_t,2)-size(kernel,2)+1);
    for con = 1:(size(x_prime_t,2)-size(kernel,2)+1)
        polych(1,con) = sum(sum(x_prime_t(:,con:(con+size(kernel,2)-1)).*kernel));
    end
    
    if disp_row(di) == 15
        slack = 4;
    else
        slack = 3;
    end
    poly_ind = find(polych >= (sum(sum(kernel))-slack));
    
    %     [I_ker,J_ker] = find(kernel);
    [I_ker,J_ker] = find(kernel(disp_ch_all,:));
    I_ker_poly = []; J_ker_poly = [];
    for i = 1:length(poly_ind)
        J_ker_poly = [J_ker_poly; J_ker-1+poly_ind(i)];
        I_ker_poly = [I_ker_poly; I_ker];
    end
    
    J_ker_poly_disp{di,1} = J_ker_poly;
    I_ker_poly_disp{di,1} = I_ker_poly;
end

% figure
% imagesc(kernel)

x_prime_t_test = x_prime_t(disp_ch_all,:);
for di = 1:length(disp_row)
    tmp_I = I_ker_poly_disp{di,1}
    tmp_J = J_ker_poly_disp{di,1}
    tmp_excld = [];
    for i = 1:length(tmp_I)
        if x_prime_t_test(tmp_I(i,1),tmp_J(i,1)) == 0
            tmp_excld = [tmp_excld i];
        end
    end
    tmp_I(tmp_excld) = [];
    tmp_J(tmp_excld) = [];
    I_ker_poly_disp{di,1} = tmp_I;
    J_ker_poly_disp{di,1} = tmp_J;
end




% load('D:\20240612_Michigan_SNN\SORN\example_N15_7ms_group')

% load('D:\20240612_Michigan_SNN\SORN\example_N24_10ms_group')

colors{1,1} = [1 0 0];
colors{2,1} = [0 0.5 0];
colors{3,1} = [0 0 1];
colors{4,1} = [0.75 0 0.75];

[x_prime_t_i, x_prime_t_j] = find(x_prime_t(disp_ch_all,:));

figure('Renderer', 'painters', 'Position', [10 10 350 170])
scatter(x_prime_t_j,x_prime_t_i,10,'.k')
for di = 1:length(disp_row)
    hold on
    jj = J_ker_poly_disp{di,1};
    ii = I_ker_poly_disp{di,1};
    scatter(jj, ii,10,colors{di,1})
end
ax = gca;
ax.FontSize = 8;
xlabel('Time (ms)', 'FontSize', 9);
ylabel('Neuron index', 'FontSize', 10);
xlim([34020 34240]); xticks([34050 34100 34150 34200]); ylim([0 100]); yticks([0 50 100])
xticklabels({'150','200','250','300'})



%
% cd E:\SORN


%% plot kernel group, red 15

disp_row = [15];

kernel = zeros(N_E,size(groups_step,2)-1);
for st = 1:(size(groups_step,2)-1)
    kernel(groups_step{disp_row,st},st) = 1;
end

[I_ker,J_ker] =  find(kernel(disp_ch_all,:))
    

figure('Renderer', 'painters', 'Position', [10 10 200 145])
scatter(J_ker-1,I_ker,10,[1 0 0],'MarkerFaceColor','w')
hold on
for i = 1:size(groups_step_conn,2)
    tmp_grp = groups_step_conn{disp_row,i}
    for j = 1:size(tmp_grp,1)
        tmp_conn = [find(disp_ch_all == tmp_grp(j,1)) find(disp_ch_all == tmp_grp(j,2))];
        plot([i-1 i],tmp_conn,'Color',[0.5 0.5 0.5])
        hold on
    end
end
scatter(J_ker-1,I_ker,10,[1 0 0],'MarkerFaceColor','w')
hold on
set(gca,'TickDir','out');
ax = gca;
ax.FontSize = 10;
xlabel('Time (ms)', 'FontSize', 11);
ylabel('Neuron index', 'FontSize', 11);
ylim([0 100]); yticks([0 50 100])

%% plot kernel group, green 12

disp_row = [12];

kernel = zeros(N_E,size(groups_step,2)-1);
for st = 1:(size(groups_step,2)-1)
    kernel(groups_step{disp_row,st},st) = 1;
end

[I_ker,J_ker] =  find(kernel(disp_ch_all,:))
    

figure('Renderer', 'painters', 'Position', [10 10 150 145])
scatter(J_ker-1,I_ker,10,[0 0.5 0],'MarkerFaceColor','w')
hold on
for i = 1:size(groups_step_conn,2)
    tmp_grp = groups_step_conn{disp_row,i}
    for j = 1:size(tmp_grp,1)
        tmp_conn = [find(disp_ch_all == tmp_grp(j,1)) find(disp_ch_all == tmp_grp(j,2))];
        plot([i-1 i],tmp_conn,'Color',[0.5 0.5 0.5])
        hold on
    end
end
scatter(J_ker-1,I_ker,10,[0 0.5 0],'MarkerFaceColor','w')
hold on
set(gca,'TickDir','out');
ax = gca;
ax.FontSize = 10;
xlabel('time (ms)', 'FontSize', 11);
ylabel('neurons', 'FontSize', 11);
ylim([0 100]); yticks([0 50 100])

%% plot kernel group, blue 22 

disp_row = [22];

kernel = zeros(N_E,size(groups_step,2)-1);
for st = 1:(size(groups_step,2)-1)
    kernel(groups_step{disp_row,st},st) = 1;
end

[I_ker,J_ker] =  find(kernel(disp_ch_all,:))
    

figure('Renderer', 'painters', 'Position', [10 10 150 145])
scatter(J_ker-1,I_ker,10,[0 0 1],'MarkerFaceColor','w')
hold on
for i = 1:size(groups_step_conn,2)
    tmp_grp = groups_step_conn{disp_row,i}
    for j = 1:size(tmp_grp,1)
        tmp_conn = [find(disp_ch_all == tmp_grp(j,1)) find(disp_ch_all == tmp_grp(j,2))];
        plot([i-1 i],tmp_conn,'Color',[0.5 0.5 0.5])
        hold on
    end
end
scatter(J_ker-1,I_ker,10,[0 0 1],'MarkerFaceColor','w')
hold on
set(gca,'TickDir','out');
ax = gca;
ax.FontSize = 10;
xlabel('time (ms)', 'FontSize', 11);
ylabel('neurons', 'FontSize', 11);
ylim([0 100]); yticks([0 50 100])
