function bic_p_q = construct_bic_p_q(bic)
%% ����bic����bic_p_q����ͬʱ����bic��ֵ���д�С��������

% ���������
% bic �� bic����

% ���������
% bic_p_q �� �����bic����

[rows,cols]= size(bic);

bic_p_q = zeros(rows*cols,3); % [bic,p,q]
for i=1:rows
   for j=1:cols
      bic_p_q((i-1)*rows+j,:)=[bic(i,j),i-1,j-1]; 
   end
end

% ����
[~,index] = sort(bic_p_q(:,1));
bic_p_q = bic_p_q(index,:);

end