clear;clc
% u = [0.5,0.5];
% a = [5,5];
% x = [0,0.5,1];
% [xx,yy] = meshgrid(x);
% z = zeros(size(xx));
% for i = 1 : 3
%     for j = 1 : 3
%         z(i,j) = gaussian([xx(i,j), yy(i,j)], u, a);
%     end
% end


[x,y] = meshgrid(-1:0.05:1);
for i = 1 : length(x)
for j = 1 : length(y)
z1(i,j) =  y(i,j);
z2(i,j) = 1* x(i,j) + y(i,j);
end
end
surf(x,y,z1); hold on
surf(x,y,z2)
xlabel('x')
ylabel('y')
zlabel('z')
legend('z1','z2')