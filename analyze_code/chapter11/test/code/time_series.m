function [ydata,p,D,q] = time_series(xdata,forecastnum,lagnum)
%% ʹ��ARIMAģ�ͽ���ʱ��Ԥ��

% ���������
% xdata�� �����ʱ�����У���������
% forecastnum�� ҪԤ��ĸ�����
% lagnum���ӳٸ�����

% ���������
% ydata�� Ԥ��Ľ��ֵ��


%% ȷ�����p��qֵ
[p,D,q,~] = find_optimal_pq(xdata,lagnum);

%% ʹ��arimaģ�ͽ���Ԥ��
ydata = arima_forecast(p,D,q,xdata,forecastnum);





end