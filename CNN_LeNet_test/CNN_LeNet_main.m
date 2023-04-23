%%%  matlab实现LeNet-5
%%%  作者：xd.wp
%%%  时间：2016.10.14  20:29
%% 程序说明
%          1、池化（pooling）采用平均2*2
%          2、网络结点数说明：
%                           输入层：28*28
%                           第一层：24*24（卷积）*6
%                           第二层：12*12（pooling）*6
%                           第三层：8*8（卷积）*16
%                           第四层：4*4（pooling）*16
%                           第五层：全连接40
%                           第六层：全连接10
%          3、网络训练部分采用800个样本，检验部分采用100个样本
clear all;clc;
%% 网络初始化
layer_c1_num=6;
layer_c2_num=16;
%权值调整步进
yita=0.05;
bias=1;
%卷积核初始化
[kernel_c1,kernel_c2]=init_kernel(layer_c1_num,layer_c2_num);
%pooling核初始化
pooling_a=ones(2,2)/4;
%全连接层的权值
weight_full_1=rand(16,40)/sqrt(40);
weight_full_2=rand(40,10)/sqrt(10);
weight_c2=rand(6,16)/10;
weight_arr2num=rand(4,4,layer_c2_num)/sqrt(16);
disp('网络初始化完成......');
%% 开始网络训练
disp('开始网络训练......');
for n=0:20
    for m=0:9
        %读取样本
        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        %         % 归一化
        %         train_data=train_data/sqrt(sum(sum(train_data.^2)));
        %标签label设置
        label_temp=-ones(1,10);
        label_temp(1,m+1)=1;
        label=label_temp;
        for iter=1:10
            %前向传递,进入卷积层1
            for k=1:layer_c1_num
                state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
                %进入pooling1
                state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
            end
            %进入卷积层2
            [state_c2,state_c2_temp]=convolution_c2(state_s1,kernel_c2,weight_c2);
            %进入pooling层2
            for k=1:layer_c2_num
                state_s2_temp1(:,:,k)=pooling(state_c2(:,:,k),pooling_a);
            end
            %将矩阵变成数
            for k=1:layer_c2_num
                state_s2_temp2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)))+bias;
                state_s2(1,k)=1/(1+exp(-state_s2_temp2(1,k)));
%             state_s2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)));
            end
            %16个特征数，进入全连接层1
            state_f1=state_s2*weight_full_1;
            %进入全连接层2
            state_f2=state_f1*weight_full_2;
            %% 误差计算部分
            Error=state_f2-label;
            Error_Cost=sum(Error.^2);
            if(Error_Cost<1e-4)
                break;
            end
            %% 参数调整部分
            [kernel_c1,kernel_c2,weight_c2,weight_full_1,weight_full_2,weight_arr2num]=CNN_upweight1(Error,train_data,...
                state_c1,state_s1,...
                state_c2,state_s2_temp1,...
                state_s2,state_s2_temp2,...
                state_f1,state_f2,...
                kernel_c1,kernel_c2,...
                weight_c2,weight_full_1,...
                weight_full_2,weight_arr2num,yita,state_c2_temp);
            
        end
    end
end
disp('网络训练完成，开始检验......');
%% 检验部分
count_num=0;
for n=10:15
    for m=0:9
        %读取样本
        train_data_test=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data_test=double(train_data_test);
        %         train_data_test=train_data_test/sqrt(sum(sum(train_data_test.^2)));
        %前向传递,进入卷积层1
        for k=1:layer_c1_num
            state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            %进入pooling1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        %进入卷积层2
        [state_c2,state_c2_temp]=convolution_c2(state_s1,kernel_c2,weight_c2);
        %进入pooling层2
        for k=1:layer_c2_num
            state_s2_temp1(:,:,k)=pooling(state_c2(:,:,k),pooling_a);
        end
        %将矩阵变成数
        for k=1:layer_c2_num
            state_s2_temp2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)))+bias;
            state_s2(1,k)=1/(1+exp(-state_s2_temp2(1,k)));
%               state_s2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)));
        end
        %16个特征数，进入全连接层1
        state_f1=state_s2*weight_full_1;
        %进入全连接层2
        state_f2=state_f1*weight_full_2;
        [~,train_label]=max(state_f2);
        if(train_label-1==m)
            count_num=count_num+1;
            train_label
        end
    end
end
ture_rate=1.0*count_num/300;

fprintf('此神经网络对MNIST样本库，识别正确率为   %4d%%    \n',ture_rate);

