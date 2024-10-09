function error = lssvm_fitness(x, p_train, t_train, p_test, t_test)
    M=size(p_train,2);
    gam = x(1);
    sig2 = x(2);
  
%%  仿真预测
 type = 'function estimation';
 [alpha,b] = trainlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel'});    %%训练模型
 t_sim =  simlssvm({p_train,t_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},p_test);
    
%%  均方根误差
error = sqrt(sum((t_sim - t_test).^2) ./ M);
    
end
