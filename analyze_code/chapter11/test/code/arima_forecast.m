function ydata = arima_forecast(p,D,q,xdata,forecastnum)
%% ARIMAʱ������ģ��

% ���������
% pvalue��AMIMAģ�͵�pֵ��
% D��AMIMAģ�͵�dֵ��
% qvalue��AMIMAģ�͵�qֵ��
% xdata�� �����ʱ�����У���������
% forecastnum�� ҪԤ��ĸ�����

% ���������
% ydata�� Ԥ��Ľ��ֵ��

Mdl = arima(p,D,q); % ����ARIMAģ�ͣ���ʼ��ģ�Ͳ���

EstMdl = estimate(Mdl,xdata); % ȷ��ģ�Ͳ���

[ydata] = forecast(EstMdl,forecastnum,'Y0',xdata);

end