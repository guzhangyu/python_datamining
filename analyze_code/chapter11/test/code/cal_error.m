function [mae_,rmse_,mape_]= cal_error(targetdata,ydata)
%% �������ٷֱ�

% ���������
% targetdata�� Ŀ��ֵ��
% ydata�� ģ�����ֵ��

% ���������
% mae_�� ƽ��������
% rmse�� ��������
% mape: ƽ�����԰ٷ���

%% �������
abs_ =abs(targetdata-ydata);
% mae
mae_=mean(abs_);
% rmse
rmse_ = mean(power(abs_,2));
% mape
mape_ = mean(abs_./targetdata);
end