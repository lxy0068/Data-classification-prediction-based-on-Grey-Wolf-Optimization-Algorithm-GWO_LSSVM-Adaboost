%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('数据集.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);
%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  创建模型
K = 10;                       % 弱分类器个数
%%  权重初始化
D = ones(1, M) / M;

%%  弱分类器
for i = 1 : K
    
%%  训练模型
SearchAgents_no=6; 
Max_iter=10;
dim=2; 
lb=[0.001,0.001];%参数下限
ub=[450,100];%参数上限
type = 'function estimation';
kernel='RBF_kernel';
[gam,sig2]=GWO(SearchAgents_no,Max_iter,lb,ub,dim,p_train,t_train,p_test,t_test);  %%优化算法
   
%%  仿真预测
 [alpha,b] = trainlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel'});    %%训练模型
 t_sim1(i, :) =  simlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},p_train);
 t_sim2(i, :) =  simlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},p_test);

%%  预测误差
    Error(i, :) = 1-sum((t_sim1(i, :)' == t_train')) / M ;
%%  调整D值
    weight(i) = 0;
    for j = 1 : M
        if abs(Error(i, j)) > 0.02
            weight(i) = weight(i) + D(i, j);
            D(i + 1, j) = D(i, j) * 1.1;
        else
            D(i + 1, j) = D(i, j);
        end
    end
    
%%  弱分类器i权重
    weight(i) = 0.5 / exp(abs(weight(i)));
    
%%  D值归一化
    D(i + 1, :) = D(i + 1, :) / sum(D(i + 1, :));
    
end

%%  强预测器预测
weight = weight / sum(weight);

%%  强分类器输出结果
T_sim1 = zeros(1, M);
T_sim2 = zeros(1, N);

for i = 1 : K
    output1 = (weight(i) * t_sim1(i, :));
    output2 = (weight(i) * t_sim2(i, :));
    
    T_sim1 = round(output1*10);
    T_sim2 = round(output2*10);
end


%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100;
error2 = sum((T_sim2 == T_test )) / N * 100;

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  混淆矩阵
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
