function [state_c2,state_c2_temp]=convolution_c2(state_s1,kernel_c2,weight_c2)
%% 完成卷积层2操作
layer_c2_num=size(weight_c2,2);
layer_s1_num=size(weight_c2,1);

%%
for n=1:layer_c2_num
    count=0;
    for m=1:layer_s1_num
        temp=state_s1(:,:,m)*weight_c2(m,n);
        count=count+temp;
    end
    state_c2_temp(:,:,n)=count;
    state_c2(:,:,n)=convolution(state_c2_temp(:,:,n),kernel_c2(:,:,n));
end
end