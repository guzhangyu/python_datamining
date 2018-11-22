function [pvalue,D,qvalue,bicvalue] = find_optimal_pq(xdata,lagnum)
%% ȷ�����p��d,qֵ

% ���������
% xdata: ԭʼ���ݣ� ������
% lagnum : �ӳٸ���

% ���������
% pvalue��AMIMAģ�͵�pֵ��
% D��AMIMAģ�͵�dֵ��
% qvalue��AMIMAģ�͵�qֵ��

%% ��ʼ��������
length_=length(xdata);
pmin=0;
pmax=round(length_/10); % һ�����������length/10
qmin=0;
qmax=round(length_/10); % һ�����������length/10

%% D ����
disp('���ڽ���D����...');
% xdata=detrend(xdata);  % ȥ������
H=adftest(xdata);
D=0; % ��ֽ���
original_data=xdata;
while ~H
    xdata=diff(xdata); % ��֣�ƽ�Ȼ�ʱ������
    D=D+1;                   % ��ִ���
    H=adftest(xdata);     % adf���飬�ж�ʱ�������Ƿ�ƽ�Ȼ�
end

%% p��q����
disp('D������ɣ����ڽ���p��q����...');
LOGL = zeros(pmax+1,qmax+1); %Initialize
PQ = zeros(pmax+1,qmax+1);

for p = pmin:pmax
    for q = qmin:qmax
        mod = arima(p,D,q);
        [~,~,logL] = estimate(mod,original_data,'print',false);
        LOGL(p+1,q+1) = logL;
        PQ(p+1,q+1) = p+q;
     end
end
% ����BIC��ֵ
LOGL = reshape(LOGL,(qmax+1)*(pmax+1),1);
PQ = reshape(PQ,(qmax+1)*(pmax+1),1);
[~,bic] = aicbic(LOGL,PQ+1,length_);
bic=reshape(bic,pmax+1,qmax+1);

% ����bic��p��q����
bic = construct_bic_p_q(bic);

% ����BICֵ��С�������в��Ƿ���ϰ�����
pvalue=-1;
qvalue=-1;
bicvalue =-1;
rows = size(bic,1);
for i= 1:rows
    if(white_noise_test(original_data,bic(i,:),D,lagnum)==1)
        bicvalue = bic(i,1);
        pvalue = bic(i,2);
        qvalue = bic(i,3);
        disp(['pֵΪ��' num2str(pvalue) ',qֵΪ��' num2str(qvalue),...
        ',BICֵΪ:' num2str(bicvalue)]);
        
        break; % ����forѭ��
    end
end
disp('p��q������ɣ�');
end

