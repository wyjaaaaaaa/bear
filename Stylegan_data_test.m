% %% ������Ȧ����С������
% clc;
% clear;
% fs=64000;                                              %����Ƶ��
% Cata1='E:\�������\��ѹ����\ȫ������';        %�ļ�����plane-Ŀ¼1
% diroutput=dir(fullfile(Cata1,'*'));                    %������ڲ�ͬ���͵��ļ����á�*����ȡ���У������ȡ�ض������ļ���'.'�����ļ����ͣ������á�.jpg��
% Catas2={diroutput.name}';                              %ȥ��������-Ŀ¼2
% Catas2=Catas2(3:length(Catas2));                       %ȥ������ǰ������еĵ�
% tic                                                    %��������ʱ�俪ʼ
% num=0;                                                 %y��ǩλ�ü���
% for gearnum=7:14 %length(Catas2)
%     Catas3{gearnum}=fullfile(Cata1,Catas2(gearnum));   %Ŀ¼3
%     DataNames=dir(fullfile(char(Catas3{gearnum}),'*'));%�����ļ�����
%     DataNames={DataNames.name}';                       %ȥ��������
%     DataNames=DataNames(3:length(DataNames));          %ȥ������ǰ������еĵ�
%     num=num+1;                                         %y��ǩλ�ü���
%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
%     
%     for i=1:length(DataNames)
% 
%         Filename=cell2mat(DataNames(i));                %Ԫ������ת���ɾ���
%         Filename=Filename(1:end-4);                     %ȥ���ļ���׺��.csv
%         Allfilename=fullfile(Catas3{gearnum},DataNames(i)); %��ȡ�ļ�Ŀ¼+�ļ���
%         load(char(Allfilename));                        %��������
%         n=2;                                            %ȡ��ʱ�����Ʋ�����n=2����0.5s��n=1����1s��
%         Vibration=eval([Filename,'.Y(7).Data']);        %ȡ��������
% 
%         for j=1:20               
%             s=Vibration((j-1)*10000+1:(j-1)*10000+38400); 
% 
%             % % ����С���任ʱƵͼ
%             wavename='cmor3-3';                        %��morletС��
%             totalscal=1024;
%             Fc=centfrq(wavename);                      % С��������Ƶ��
%             c=2*Fc*totalscal;
%             scals=c./(1:totalscal);
%             w1=abs(cwt(s,scals,wavename));            % �õ�����С��ϵ��
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
%             m=(gearnum-7)*80*20+(i-1)*20+j;                         %��������
%             coefs=coefs(end:-1:1,:);
%             x(m,:,:)=coefs*70;           
%             
% %             y(m)=Lab;                     %�洢y��ǩ                
%             toc                           %��������ʱ��
%             clear -regexp ^N
%             disp(['����������',num2str(m),'��']);
%         end
%     end
% end
% save('E:\�������\Stylegan_WT_OR_data.mat','x','-v7.3');             %����ʱƵ����
% 
% %% ���ɻҶ�ͼ
% clc;clear;close;
% load('E:\�������\Stylegan_WT_OR_data.mat')
% for i=1:length(x)  
%     imshow(uint8(reshape(x(i,:,:),256,256)),'border','tight');
%     colormap(flipud(gray));
%     f=getframe;
%     imwrite(f.cdata,['E:\�������\Stylegan_OR_map\',num2str(i),'.png']); %�����figure���ڣ��ߴ���ֱ��ʲ���
% end


%% ���ɲ��Լ�����С������
clc;
clear;
fs=64000;                                              %����Ƶ��
Cata1='E:\�������\��ѹ����\������test����';        %�ļ�����plane-Ŀ¼1
diroutput=dir(fullfile(Cata1,'*'));                    %������ڲ�ͬ���͵��ļ����á�*����ȡ���У������ȡ�ض������ļ���'.'�����ļ����ͣ������á�.jpg��
Catas2={diroutput.name}';                              %ȥ��������-Ŀ¼2
Catas2=Catas2(3:length(Catas2));                       %ȥ������ǰ������еĵ�
tic                                                    %��������ʱ�俪ʼ
num=0;                                                 %y��ǩλ�ü���
for gearnum=1:12 %length(Catas2)
    Catas3{gearnum}=fullfile(Cata1,Catas2(gearnum));   %Ŀ¼3
    DataNames=dir(fullfile(char(Catas3{gearnum}),'*'));%�����ļ�����
    DataNames={DataNames.name}';                       %ȥ��������
    DataNames=DataNames(3:length(DataNames));          %ȥ������ǰ������еĵ�
    num=num+1;                                         %y��ǩλ�ü���
                   
    
    for i=1:length(DataNames)

        Filename=cell2mat(DataNames(i));                %Ԫ������ת���ɾ���
        Filename=Filename(1:end-4);                     %ȥ���ļ���׺��.csv
        Allfilename=fullfile(Catas3{gearnum},DataNames(i)); %��ȡ�ļ�Ŀ¼+�ļ���
        load(char(Allfilename));                        %��������
        n=2;                                            %ȡ��ʱ�����Ʋ�����n=2����0.5s��n=1����1s��
        Vibration=eval([Filename,'.Y(7).Data']);        %ȡ��������

        for j=1:10               
            s=Vibration((j-1)*10000+1:(j-1)*10000+38400); 

            % % ����С���任ʱƵͼ
            wavename='cmor3-3';                        %��morletС��
            totalscal=1024;
            Fc=centfrq(wavename);                      % С��������Ƶ��
            c=2*Fc*totalscal;
            scals=c./(1:totalscal);
            w1=abs(cwt(s,scals,wavename));            % �õ�����С��ϵ��
       
            q1=w1(1:512,:);

            for u=1:size(q1,1)
                e1(u,:)=downsample(q1(u,:),150);

            end
            for v=1:size(e1,2)
                coefs(:,v,1)=downsample(e1(:,v),2);
           
            end              
            m=(gearnum-1)*80*10+(i-1)*10+j;                         %��������
            coefs=coefs(end:-1:1,:);
            x(m,:,:)=coefs*70;           
            
%             y(m)=Lab;                     %�洢y��ǩ                
            toc                           %��������ʱ��
            clear -regexp ^N
            disp(['����������',num2str(m),'��']);
        end
    end
end
save('E:\�������\Stylegan_WT_test_all.mat','x','-v7.3');             %����ʱƵ����

%% ���ɻҶ�ͼ
clc;clear;close;
load('E:\�������\Stylegan_WT_test_all.mat')
for i=1:length(x)  
    imshow(uint8(reshape(x(i,:,:),256,256)),'border','tight');
    colormap(flipud(gray));
    f=getframe;
    imwrite(f.cdata,['E:\�������\Stylegan_test_all\',num2str(i),'.png']); %�����figure���ڣ��ߴ���ֱ��ʲ���
end