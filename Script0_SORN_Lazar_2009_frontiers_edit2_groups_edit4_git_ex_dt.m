
clear all 
% load('D:\20240612_Michigan_SNN\SORN\example_N15_7ms_group')

 load('D:\20240612_Michigan_SNN\SORN\example_N24_10ms_group')

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
