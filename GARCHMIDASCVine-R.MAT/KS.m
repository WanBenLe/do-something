load stockreturns
x = stocks(:,1);
y = stocks(:,2);
u = ksdensity(x,x,'function','cdf');
v = ksdensity(y,y,'function','cdf');
a=v*0.1+0.5
[Rho,nu] = copulafita('t',[u v a],'Method','ApproximateML')
rho = copulaparam('t',Rho)
das = copularnd('t',Rho,nu,100) 