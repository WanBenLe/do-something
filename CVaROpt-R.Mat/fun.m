function outputArg1 = fun(x)
data=readtable('COOLR.csv');
f=data{:,2:7};
r=x(1)*f(:,1)+x(2)*f(:,2)+x(3)*f(:,3)+x(4)*f(:,4)+x(5)*f(:,5)+x(6)*f(:,6);
XX=sum(r);
outputArg1(1)=-XX;

%网格数
grid=10000;
var=zeros(grid+1,1);
parfor i=1:(grid+1)
    var(i)=quantile(r,(i-1)/grid);
end
CVaR=5;
%百分位点
alpha=1;
parfor i=1:floor(alpha/100*grid)
    CVaR=CVaR-(((var(i)+var(i+1))/2*(1/grid))*100/alpha);
end
% if CVaR>0
%     CVaR=0;
% end

outputArg1(2)=-CVaR;

disp(XX)
disp(CVaR)
end

