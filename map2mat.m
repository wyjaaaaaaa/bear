% %% ����OR����
% clc;close all;clear 
% j=0;
% for i=1:2:3000
%     j=j+1
%     x(j,:,:,:)=imread(['E:\Stlyegan\stylegan-master\results\Generate_OR\',num2str(i-1),'.png']);  
%    end
% save('E:\�������\MAT����\Stylegan_Generate_OR_data.mat','x','-v7.3'); 

%% ԭʼOR����
% clc;close all;clear 
% j=0
% for i=1:8:12000
%     j=j+1
%     x(j,:,:,:)=imread(['E:\�������\Stylegan_OR_map\',num2str(i),'.png']);  
%    
% end
% save('E:\�������\Resnet_data\Stylegan_Real_OR_data.mat','x','-v7.3'); 

%% ��������
clc;close all;clear 
j=0
for i=1:5:9600
   j=j+1
    x(j,:,:,:)=imread(['E:\�������\Stylegan_Test_map\',num2str(i),'.png']);  
   
end
save('E:\�������\Resnet_data\Stylegan_Test_data.mat','x','-v7.3'); 

% %% IR����
% clc;close all;clear 
% j=0
% for i=1:6:9000
%    j=j+1
%     x(j,:,:,:)=imread(['E:\�������\Stylegan_IR_map\',num2str(i),'.png']);  
%    
% end
% save('E:\�������\Resnet_data\Stylegan_IR_data.mat','x','-v7.3');
% 
% %% ��������
% clc;close all;clear 
% j=0
% for i=1:2:3000
%    j=j+1
%     x(j,:,:,:)=imread(['E:\�������\Stylegan_Normal_map\',num2str(i),'.png']);  
%    
% end
% save('E:\�������\Resnet_data\Stylegan_Normal_data.mat','x','-v7.3');
