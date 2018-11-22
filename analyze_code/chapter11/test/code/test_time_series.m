%% time_series���Դ���
clear;
% ��ʼ��������
filename='../data/time_data.xls';
index = 2;
forecastnum =5; % ҪԤ��ĸ�����
lagnum =12; % �ӳٸ���
outputfile = '../tmp/forecast.xls';

%% ��ȡ����
data = xlsread(filename);
xdata=data(1:end-forecastnum,index);   % �����ʱ�����У���������
% xdata=xdata/1024/1024; % ת�����ݸ�ʽ��


%% ���� arima �㷨���в���
[ydata,p,D,q] = time_series(xdata,forecastnum,lagnum);

disp(['ARIMAģ�͵�p��D��qֵ�ֱ�Ϊ��' num2str(p) ',' num2str(D) ',' num2str(q)]);
disp(['Ԥ��ֵΪ��' num2str(ydata')]);

%% �������ٷֱȣ���������д���ļ�
targetdata=data(end-forecastnum+1:end,index);
% ���ݸ�ʽת��
targetdata=targetdata/1024/1024;
ydata =ydata/1024/1024;
% �������
[mae_,rmse_,mape_]= cal_error(targetdata,ydata);
xlswrite(outputfile,[{'id','Ԥ��ֵ','ʵ��ֵ'};...
    num2cell([[1:forecastnum]',ydata,targetdata])]);
disp(['ƽ��������' num2str(mae_) ', ��������' num2str(rmse_) ...
    ', ƽ�����԰ٷ���' num2str(mape_)]);
disp('ʱ�����в�����ɣ�');