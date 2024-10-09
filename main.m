%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��ȡ����
res = xlsread('���ݼ�.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);
%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  ת������Ӧģ��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  ����ģ��
K = 10;                       % ������������
%%  Ȩ�س�ʼ��
D = ones(1, M) / M;

%%  ��������
for i = 1 : K
    
%%  ѵ��ģ��
SearchAgents_no=6; 
Max_iter=10;
dim=2; 
lb=[0.001,0.001];%��������
ub=[450,100];%��������
type = 'function estimation';
kernel='RBF_kernel';
[gam,sig2]=GWO(SearchAgents_no,Max_iter,lb,ub,dim,p_train,t_train,p_test,t_test);  %%�Ż��㷨
   
%%  ����Ԥ��
 [alpha,b] = trainlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel'});    %%ѵ��ģ��
 t_sim1(i, :) =  simlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},p_train);
 t_sim2(i, :) =  simlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},p_test);

%%  Ԥ�����
    Error(i, :) = 1-sum((t_sim1(i, :)' == t_train')) / M ;
%%  ����Dֵ
    weight(i) = 0;
    for j = 1 : M
        if abs(Error(i, j)) > 0.02
            weight(i) = weight(i) + D(i, j);
            D(i + 1, j) = D(i, j) * 1.1;
        else
            D(i + 1, j) = D(i, j);
        end
    end
    
%%  ��������iȨ��
    weight(i) = 0.5 / exp(abs(weight(i)));
    
%%  Dֵ��һ��
    D(i + 1, :) = D(i + 1, :) / sum(D(i + 1, :));
    
end

%%  ǿԤ����Ԥ��
weight = weight / sum(weight);

%%  ǿ������������
T_sim1 = zeros(1, M);
T_sim2 = zeros(1, N);

for i = 1 : K
    output1 = (weight(i) * t_sim1(i, :));
    output2 = (weight(i) * t_sim2(i, :));
    
    T_sim1 = round(output1*10);
    T_sim2 = round(output2*10);
end


%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100;
error2 = sum((T_sim2 == T_test )) / N * 100;

%%  ��������
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid

%%  ��������
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
