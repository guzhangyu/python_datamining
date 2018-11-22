function [ result ] = white_noise_test(data, bic,D,lagnum )
%% ����������

% �������ݣ�
% data: ԭʼ���ݼ�
% bic��[bic,p,q]ֵ�����
% D����ֽ״�
% lagnum : �ӳٸ���

% ���������
% result���Ƿ���ϰ��������飬1��ʾ���ϣ�0��ʾ������

%% ����ģ��
mod = arima(bic(1,2),D,bic(1,3));
[EstMdl,~,~] = estimate(mod,data,'print',false);
% ����в�
res = infer(EstMdl,data);
stdRes = res/sqrt(EstMdl.Variance); % ��׼���в�
% ����������
[h,~] = lbqtest(stdRes,'lags',1:lagnum);
% ���㲻���ϰ���������ĸ���
hsum = sum(h);
if hsum==0;
    result =1; % ���ϰ���������
else
    result=0; % �����ϰ���������
end
    
end


