% %% 生成外圈故障小波数据
% clc;
% clear;
% fs=64000;                                              %采样频率
% Cata1='E:\轴承数据\解压数据\全部数据';        %文件夹名plane-目录1
% diroutput=dir(fullfile(Cata1,'*'));                    %如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
% Catas2={diroutput.name}';                              %去掉单引号-目录2
% Catas2=Catas2(3:length(Catas2));                       %去掉胞组前面的两行的点
% tic                                                    %样本计算时间开始
% num=0;                                                 %y标签位置计数
% for gearnum=7:14 %length(Catas2)
%     Catas3{gearnum}=fullfile(Cata1,Catas2(gearnum));   %目录3
%     DataNames=dir(fullfile(char(Catas3{gearnum}),'*'));%数据文件名称
%     DataNames={DataNames.name}';                       %去掉单引号
%     DataNames=DataNames(3:length(DataNames));          %去掉胞组前面的两行的点
%     num=num+1;                                         %y标签位置计数
%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
%     
%     for i=1:length(DataNames)
% 
%         Filename=cell2mat(DataNames(i));                %元胞数组转换成矩阵
%         Filename=Filename(1:end-4);                     %去掉文件后缀名.csv
%         Allfilename=fullfile(Catas3{gearnum},DataNames(i)); %获取文件目录+文件名
%         load(char(Allfilename));                        %加载数据
%         n=2;                                            %取样时长控制参数（n=2代表0.5s，n=1代表1s）
%         Vibration=eval([Filename,'.Y(7).Data']);        %取出振动数据
% 
%         for j=1:20               
%             s=Vibration((j-1)*10000+1:(j-1)*10000+38400); 
% 
%             % % 连续小波变换时频图
%             wavename='cmor3-3';                        %复morlet小波
%             totalscal=1024;
%             Fc=centfrq(wavename);                      % 小波的中心频率
%             c=2*Fc*totalscal;
%             scals=c./(1:totalscal);
%             w1=abs(cwt(s,scals,wavename));            % 得到连续小波系数
%        
%             q1=w1(1:512,:);
% 
%             for u=1:size(q1,1)
%                 e1(u,:)=downsample(q1(u,:),150);
% 
%             end
%             for v=1:size(e1,2)
%                 coefs(:,v,1)=downsample(e1(:,v),2);
%            
%             end              
%             m=(gearnum-7)*80*20+(i-1)*20+j;                         %样本计数
%             coefs=coefs(end:-1:1,:);
%             x(m,:,:)=coefs*70;           
%             
% %             y(m)=Lab;                     %存储y标签                
%             toc                           %样本计算时间
%             clear -regexp ^N
%             disp(['样本计数：',num2str(m),'个']);
%         end
%     end
% end
% save('E:\轴承数据\Stylegan_WT_OR_data.mat','x','-v7.3');             %保存时频数据
% 
% %% 生成灰度图
% clc;clear;close;
% load('E:\轴承数据\Stylegan_WT_OR_data.mat')
% for i=1:length(x)  
%     imshow(uint8(reshape(x(i,:,:),256,256)),'border','tight');
%     colormap(flipud(gray));
%     f=getframe;
%     imwrite(f.cdata,['E:\轴承数据\Stylegan_OR_map\',num2str(i),'.png']); %保存的figure窗口，尺寸与分辨率不变
% end


%% 生成测试集故障小波数据
clc;
clear;
fs=64000;                                              %采样频率
Cata1='E:\轴承数据\解压数据\加速损坏test数据';        %文件夹名plane-目录1
diroutput=dir(fullfile(Cata1,'*'));                    %如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
Catas2={diroutput.name}';                              %去掉单引号-目录2
Catas2=Catas2(3:length(Catas2));                       %去掉胞组前面的两行的点
tic                                                    %样本计算时间开始
num=0;                                                 %y标签位置计数
for gearnum=1:12 %length(Catas2)
    Catas3{gearnum}=fullfile(Cata1,Catas2(gearnum));   %目录3
    DataNames=dir(fullfile(char(Catas3{gearnum}),'*'));%数据文件名称
    DataNames={DataNames.name}';                       %去掉单引号
    DataNames=DataNames(3:length(DataNames));          %去掉胞组前面的两行的点
    num=num+1;                                         %y标签位置计数
                   
    
    for i=1:length(DataNames)

        Filename=cell2mat(DataNames(i));                %元胞数组转换成矩阵
        Filename=Filename(1:end-4);                     %去掉文件后缀名.csv
        Allfilename=fullfile(Catas3{gearnum},DataNames(i)); %获取文件目录+文件名
        load(char(Allfilename));                        %加载数据
        n=2;                                            %取样时长控制参数（n=2代表0.5s，n=1代表1s）
        Vibration=eval([Filename,'.Y(7).Data']);        %取出振动数据

        for j=1:10               
            s=Vibration((j-1)*10000+1:(j-1)*10000+38400); 

            % % 连续小波变换时频图
            wavename='cmor3-3';                        %复morlet小波
            totalscal=1024;
            Fc=centfrq(wavename);                      % 小波的中心频率
            c=2*Fc*totalscal;
            scals=c./(1:totalscal);
            w1=abs(cwt(s,scals,wavename));            % 得到连续小波系数
       
            q1=w1(1:512,:);

            for u=1:size(q1,1)
                e1(u,:)=downsample(q1(u,:),150);

            end
            for v=1:size(e1,2)
                coefs(:,v,1)=downsample(e1(:,v),2);
           
            end              
            m=(gearnum-1)*80*10+(i-1)*10+j;                         %样本计数
            coefs=coefs(end:-1:1,:);
            x(m,:,:)=coefs*70;           
            
%             y(m)=Lab;                     %存储y标签                
            toc                           %样本计算时间
            clear -regexp ^N
            disp(['样本计数：',num2str(m),'个']);
        end
    end
end
save('E:\轴承数据\Stylegan_WT_test_all.mat','x','-v7.3');             %保存时频数据

%% 生成灰度图
clc;clear;close;
load('E:\轴承数据\Stylegan_WT_test_all.mat')
for i=1:length(x)  
    imshow(uint8(reshape(x(i,:,:),256,256)),'border','tight');
    colormap(flipud(gray));
    f=getframe;
    imwrite(f.cdata,['E:\轴承数据\Stylegan_test_all\',num2str(i),'.png']); %保存的figure窗口，尺寸与分辨率不变
end