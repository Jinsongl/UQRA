[max_extrm, max_idx] = max(qoi_median);
plot(X(1,:),X(2,:),'.k');
hold on
scatter(X(1,:),X(2,:),5,qoi_median);

plot(X(1,max_idx),X(2,max_idx),'or');
grid on


