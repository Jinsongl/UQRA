
% S_FLF=S_FLF(1:2,1:2,:);

fac1=0.585809078570351;fac2=1;

Diag_surge=Diag_surge*fac1;  %0.255554093091961
% Diag_sway=Diag_sway*fac2;

TFv=TFv([1,2],:);

% -----------
% Mass matrix
% -----------

M(1,1)=1e8;
M(2,2)=1.4e8;  %M(2,2)=4e8;

% ----------------
% Stiffness matrix
% ----------------

K=[280000 0;0 280000];

% --------------
% Damping matrix
% --------------

DR=.05;
B(1,1)=DR*2*sqrt(K(1,1)*M(1,1));
B(2,2)=DR*2*sqrt(K(2,2)*M(2,2));

% -----------
% WF analysis
% -----------

for xx=1:N
RAO(:,xx)=(-w(xx)^2*M+i*w(xx)*B+K)\TFv(:,xx);
end
