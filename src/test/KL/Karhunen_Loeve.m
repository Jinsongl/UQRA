close all
s = linspace(0,1,200);
t = s;
[ss,tt] = meshgrid(s,t);
R = exp(-abs(ss-tt));
surf(ss,tt,R)
[V,D] = eig(R);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
evalues = diag(Ds);
evalues_sum = sum(evalues);
figure
for i = 1 : 10 
    plot(Vs(:,i)*sqrt(Ds(i,i))); hold on
end
figure


plot(evalues(1:10)./sum(evalues),'.');
N = 6;
RN = 0 * R;
for is = 1 : length(s)
    for it = 1 : length(t)
        for in = 1 : N
            RN(is, it) = RN(is, it) + evalues(in) * Vs(is,in) * Vs(it, in);
        end
    end
end
figure
surf(ss,tt,RN)

RN_error = R - RN;
figure
surf(ss,tt,RN_error)

        