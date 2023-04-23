%%%  matlabʵ��LeNet-5
%%%  ���ߣ�xd.wp
%%%  ʱ�䣺2016.10.14  20:29
%% ����˵��
%          1���ػ���pooling������ƽ��2*2
%          2����������˵����
%                           ����㣺28*28
%                           ��һ�㣺24*24�������*6
%                           �ڶ��㣺12*12��pooling��*6
%                           �����㣺8*8�������*16
%                           ���Ĳ㣺4*4��pooling��*16
%                           ����㣺ȫ����40
%                           �����㣺ȫ����10
%          3������ѵ�����ֲ���800�����������鲿�ֲ���100������
clear all;clc;
%% �����ʼ��
layer_c1_num=6;
layer_c2_num=16;
%Ȩֵ��������
yita=0.05;
bias=1;
%����˳�ʼ��
[kernel_c1,kernel_c2]=init_kernel(layer_c1_num,layer_c2_num);
%pooling�˳�ʼ��
pooling_a=ones(2,2)/4;
%ȫ���Ӳ��Ȩֵ
weight_full_1=rand(16,40)/sqrt(40);
weight_full_2=rand(40,10)/sqrt(10);
weight_c2=rand(6,16)/10;
weight_arr2num=rand(4,4,layer_c2_num)/sqrt(16);
disp('�����ʼ�����......');
%% ��ʼ����ѵ��
disp('��ʼ����ѵ��......');
for n=0:20
    for m=0:9
        %��ȡ����
        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        %         % ��һ��
        %         train_data=train_data/sqrt(sum(sum(train_data.^2)));
        %��ǩlabel����
        label_temp=-ones(1,10);
        label_temp(1,m+1)=1;
        label=label_temp;
        for iter=1:10
            %ǰ�򴫵�,��������1
            for k=1:layer_c1_num
                state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
                %����pooling1
                state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
            end
            %��������2
            [state_c2,state_c2_temp]=convolution_c2(state_s1,kernel_c2,weight_c2);
            %����pooling��2
            for k=1:layer_c2_num
                state_s2_temp1(:,:,k)=pooling(state_c2(:,:,k),pooling_a);
            end
            %����������
            for k=1:layer_c2_num
                state_s2_temp2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)))+bias;
                state_s2(1,k)=1/(1+exp(-state_s2_temp2(1,k)));
%             state_s2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)));
            end
            %16��������������ȫ���Ӳ�1
            state_f1=state_s2*weight_full_1;
            %����ȫ���Ӳ�2
            state_f2=state_f1*weight_full_2;
            %% �����㲿��
            Error=state_f2-label;
            Error_Cost=sum(Error.^2);
            if(Error_Cost<1e-4)
                break;
            end
            %% ������������
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
disp('����ѵ����ɣ���ʼ����......');
%% ���鲿��
count_num=0;
for n=10:15
    for m=0:9
        %��ȡ����
        train_data_test=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data_test=double(train_data_test);
        %         train_data_test=train_data_test/sqrt(sum(sum(train_data_test.^2)));
        %ǰ�򴫵�,��������1
        for k=1:layer_c1_num
            state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            %����pooling1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        %��������2
        [state_c2,state_c2_temp]=convolution_c2(state_s1,kernel_c2,weight_c2);
        %����pooling��2
        for k=1:layer_c2_num
            state_s2_temp1(:,:,k)=pooling(state_c2(:,:,k),pooling_a);
        end
        %����������
        for k=1:layer_c2_num
            state_s2_temp2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)))+bias;
            state_s2(1,k)=1/(1+exp(-state_s2_temp2(1,k)));
%               state_s2(1,k)=sum(sum(state_s2_temp1(:,:,k).*weight_arr2num(:,:,k)));
        end
        %16��������������ȫ���Ӳ�1
        state_f1=state_s2*weight_full_1;
        %����ȫ���Ӳ�2
        state_f2=state_f1*weight_full_2;
        [~,train_label]=max(state_f2);
        if(train_label-1==m)
            count_num=count_num+1;
            train_label
        end
    end
end
ture_rate=1.0*count_num/300;

fprintf('���������MNIST�����⣬ʶ����ȷ��Ϊ   %4d%%    \n',ture_rate);

