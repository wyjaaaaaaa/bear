% %% 生成OR数据
% clc;close all;clear 
% j=0;
% for i=1:2:3000
%     j=j+1
%     x(j,:,:,:)=imread(['E:\Stlyegan\stylegan-master\results\Generate_OR\',num2str(i-1),'.png']);  
%    end
% save('E:\轴承数据\MAT数据\Stylegan_Generate_OR_data.mat','x','-v7.3'); 

%% 原始OR数据
% clc;close all;clear 
% j=0
% for i=1:8:12000
%     j=j+1
%     x(j,:,:,:)=imread(['E:\轴承数据\Stylegan_OR_map\',num2str(i),'.png']);  
%    
% end
% save('E:\轴承数据\Resnet_data\Stylegan_Real_OR_data.mat','x','-v7.3'); 

%% 测试数据
clc;close all;clear 
j=0
for i=1:5:9600
   j=j+1
    x(j,:,:,:)=imread(['E:\轴承数据\Stylegan_Test_map\',num2str(i),'.png']);  
   
end
save('E:\轴承数据\Resnet_data\Stylegan_Test_data.mat','x','-v7.3'); 

% %% IR数据
% clc;close all;clear 
% j=0
% for i=1:6:9000
%    j=j+1
%     x(j,:,:,:)=imread(['E:\轴承数据\Stylegan_IR_map\',num2str(i),'.png']);  
%    
% end
% save('E:\轴承数据\Resnet_data\Stylegan_IR_data.mat','x','-v7.3');
% 
% %% 正常数据
% clc;close all;clear 
% j=0
% for i=1:2:3000
%    j=j+1
%     x(j,:,:,:)=imread(['E:\轴承数据\Stylegan_Normal_map\',num2str(i),'.png']);  
%    
% end
% save('E:\轴承数据\Resnet_data\Stylegan_Normal_data.mat','x','-v7.3');
