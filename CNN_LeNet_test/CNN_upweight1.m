function [kernel_c1,kernel_c2,weight_c2,weight_full_1,weight_full_2,weight_arr2num]=CNN_upweight1(Error,train_data,state_c1,state_s1,state_c2,state_s2_temp1,state_s2_temp2,state_s2,...
                                                                                                state_f1,state_f2,kernel_c1,kernel_c2,weight_c2,weight_full_1,weight_full_2,weight_arr2num,yita,state_c2_temp1)
%%%     完成参数更新，权值和卷积核
%% 结点数目
layer_f2_num=size(state_f2,2);
layer_f1_num=size(state_f1,2);
layer_c2_num=size(state_c2,3);
layer_c1_num=size(state_c1,3);
[s2_row,s2_col,~]=size(state_s2_temp1);
[kernel_row,kernel_col]=size(kernel_c1(:,:,1));
%% 保存网络权值
weight_full_2_temp=weight_full_2;
weight_full_1_temp=weight_full_1;
weight_arr2num_temp=weight_arr2num;
kernel_c2_temp=kernel_c2;
kernel_c1_temp=kernel_c1;
weight_c2_temp=weight_c2;
%% 更新weight_full_2
for n=1:layer_f2_num
    delta_weight_full_2_temp(:,n)=2*Error(1,n)*state_f1';
end
size(delta_weight_full_2_temp);
weight_full_2_temp=weight_full_2_temp-yita*delta_weight_full_2_temp;

%% 更新weight_full_1
for n=1:layer_f2_num
    for m=1:layer_f1_num
        delta_weight_full_1_temp(:,m)=2*Error(1,n)*weight_full_2(m,n)*state_s2';
    end
    weight_full_1_temp=weight_full_1_temp-yita*delta_weight_full_1_temp;
end
%% 更新weight_arr2num
for m=1:layer_c2_num
    for n=1:layer_f2_num
        count_delta_weight_arr2num_temp=zeros(s2_row,s2_col);
        for k=1:layer_f1_num
            delta_weight_arr2num_temp(:,:,m)=2*Error(1,n)*weight_full_1(m,k)*weight_full_2(k,n)*exp(-state_s2_temp2(1,m))/(1+exp(-state_s2_temp2(1,m))).^2*state_s2_temp1(:,:,m);
            count_delta_weight_arr2num_temp=count_delta_weight_arr2num_temp+delta_weight_arr2num_temp(:,:,m);
        end
        weight_arr2num_temp(:,:,m)=weight_arr2num_temp(:,:,m)-yita*count_delta_weight_arr2num_temp;
    end
end
%% 更新kernel_c2
for m=1:layer_c2_num
    for n=1:layer_f2_num
        count_delta_state_s2_temp1=zeros(s2_row,s2_col);
        for k=1:layer_f1_num
            delta_state_s2_temp1(:,:,m)=2*Error(1,n)*weight_full_1(m,k)*weight_full_2(k,n)*exp(-state_s2_temp2(1,m))/(1+exp(-state_s2_temp2(1,m))).^2*weight_arr2num(:,:,m);
            count_delta_state_s2_temp1=count_delta_state_s2_temp1+delta_state_s2_temp1(:,:,m);
        end
            delta_state_c2=kron(count_delta_state_s2_temp1,ones(2,2)/4);
            count=state_c2_temp1(1:kernel_row,1:kernel_col)*delta_state_c2(1,1);
        kernel_c2_temp(:,:,m)=kernel_c2_temp(:,:,m)-yita*count;
    end
end

%% 更新 weight_c2
for n=1:layer_f2_num
    for m=1:layer_c2_num
        count_delta_state_s2_temp1=zeros(s2_row,s2_col);
        for kk=1:layer_f1_num
             delta_state_s2_temp1(:,:,m)=2*Error(1,n)*weight_full_1(m,kk)*weight_full_2(kk,n)*exp(-state_s2_temp2(1,m))/(1+exp(-state_s2_temp2(1,m))).^2*weight_arr2num(:,:,m);
             count_delta_state_s2_temp1=count_delta_state_s2_temp1+delta_state_s2_temp1(:,:,m);
        end
        delta_state_c2=kron(count_delta_state_s2_temp1,ones(2,2)/4);
        delta_state_c2_temp1(:,:,m)=conv2(delta_state_c2,kernel_c2(:,:,m),'full');
        for k=1:layer_c1_num
            delta_weight_c2_temp(k,m)=sum(sum(delta_state_c2_temp1(:,:,m).*state_s1(:,:,k)));
        end
    end
    weight_c2_temp=weight_c2_temp-yita*delta_weight_c2_temp;
end

%% 更新 kernel_c1
for n=1:layer_f2_num
    for m=1:layer_c2_num
        count_delta_state_s2_temp1=zeros(s2_row,s2_col);
        for kk=1:layer_f1_num
            delta_state_s2_temp1(:,:,m)=2*Error(1,n)*weight_full_1(m,kk)*weight_full_2(kk,n)*exp(-state_s2_temp2(1,m))/(1+exp(-state_s2_temp2(1,m))).^2*weight_arr2num(:,:,m);
            count_delta_state_s2_temp1=count_delta_state_s2_temp1+delta_state_s2_temp1(:,:,m);
        end
        delta_state_c2=kron(count_delta_state_s2_temp1,ones(2,2)/4);
        delta_state_c2_temp1(:,:,m)=conv2(delta_state_c2,kernel_c2(:,:,m),'full');
    end
    for k=1:layer_c1_num
        [x,y,~]=size(state_c2_temp1);
        count1=zeros(x,y);
        for m=1:layer_c2_num
            count_temp=weight_c2(k,m)*delta_state_c2_temp1(:,:,m);
            count1=count1+count_temp;
        end
        delta_state_s1(:,:,k)=count1;
        delta_state_c1(:,:,k)=kron(delta_state_s1(:,:,k),ones(2,2)/4);
        %
        count2=train_data(1:kernel_row,1:kernel_col)*delta_state_c1(1,1,k);
        kernel_c1_temp(:,:,k)=kernel_c1_temp(:,:,k)-yita*count2;
    end
    
end

%% 权值更新
weight_c2=weight_c2_temp;
kernel_c1=kernel_c1_temp;
kernel_c2=kernel_c2_temp;
weight_arr2num=weight_arr2num_temp;
weight_full_2=weight_full_2_temp;
weight_full_1=weight_full_1_temp;

end