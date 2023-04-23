function [kernel_c1,kernel_c2]=init_kernel(layer_c1_num,layer_c2_num)
%% ¾í»ıºË³õÊ¼»¯
for n=1:layer_c1_num
    kernel_c1(:,:,n)=rand(5,5)/sqrt(6);
end
for n=1:layer_c2_num
    kernel_c2(:,:,n)=rand(5,5)/sqrt(16);
end


end