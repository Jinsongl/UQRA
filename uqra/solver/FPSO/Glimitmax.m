function Gmax=Glimitmax(xin)

global Diag_surge N N_LF Snn TFv dw dw_LF w w_LF RAO wmin wmax Hx


% ------------------
% Environmental data
% ------------------
Hs=xin(1);
Tp=xin(2);

Snn=Jonswap(Hs,Tp);

N=960;
Nt=125664;




% ----------
% Parameters
% ----------

Re=xin(3:962);
Im=xin(963:1922);
A=(Re+i*Im).*sqrt(dw*Snn);
X_0 =-Nt*real(ifft(A,Nt,2));
% sigma = std(X_0) * 4
% -----------
% WF response
% -----------

Z=[zeros(1,wmin/dw),RAO(1,:).*A];


% -----------
% LF response
% -----------

for xx=160:-1:1
A_Aconj=A(1:N-xx).*conj(A(xx+1:N));
X1(xx+1)=Hx(xx+1)*sum(0.5*(Diag_surge(1:N-xx)+Diag_surge(xx+1:N)).*A_Aconj);
end

Z(1:length(X1))=Z(1:length(X1))+2*X1;

X=-Nt*real(ifft(Z,Nt,2));

Gmax=max(X(1:41888));

%[Hs Tp]
% plot(X(1:41888))


