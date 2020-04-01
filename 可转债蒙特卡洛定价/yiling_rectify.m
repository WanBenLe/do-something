function [a,Price]=yiling_rectify(cp,X,coupon,mcallshedule,sigma,r,T)
%Dara是一列利率数据
Data=xlsread('data.xlsx');
x=Data(1:end-1);
TimeStep=1/250;
dx=diff (Data);
dx=dx./x.^0.5;
regr=[TimeStep./x.^0.5,  TimeStep*x.^0.5];
drift=regr\dx;
res=regr*drift-dx;
alpha=-drift (2) ;
R=-drift (1) /drift (2);
sigma=sqrt (var (res, 1) /TimeStep);
InitialParams=[alpha R sigma]
 
IimeStep=1/250;
x=Data(1:end-1);
dx=diff(Data);
dx=dx./x.^0.5;
regr=[TimeStep./x.^0.5,TimeStep*x.^0.5];
drift=regr\dx;
res=regr*drift-dx;
alpha=-drift(2);
R=-drift(1)/drift(2);
sigma=sqrt(var(res, 1)/TimeStep);
InitialParams=[alpha R sigma];
DataA=Data(2:end);
DataB=Data(1 :end-1);
Leng=length(Data);
c=@(Params)2*Params(1)/(Params(3)^2*(1-exp(-Params(1)*TimeStep)));
q=@(Params)2*Params(1)*Params(2)/Params(3)^2-1;
u=@(Params)c(Params)*exp(-Params(1)*TimeStep)*DataB;
v=@(Params)c(Params)*DataA;
z=@(Params)2*sqrt(u(Params).*v(Params));
bf=@(Params)besseli(q(Params),z(Params),1);
lnL=@(Params)-(Leng-1)*log(c(Params))+sum(u(Params)+v(Params)-0.5*q(Params)*log(v(Params)./u(Params))-log(bf(Params))-z(Params));
[Params,Pval]=fminsearch(lnL,[0.1,0.01,0.1])




p1=8/252;%涨停
p2=6/252;%跌停
lamuda=4*6/252/2;%季度因为观测2年除以2
cp=16.08;%初始股价
X=13.69;%转股价
coupon=[0.002,0.004,0.006,0.01,0.015,0.020]; %利率
mcallshedule=106;%到期赎回价
sigma=0.37;%方差，股票的波动率
r=0.0342;%市场无风险率
T=6;%需要观察几年
Nstep=T*360; %总模拟的时间 T*360天一年12月一个假设30天
Nrepl=1000;%路径条数
dt=T/Nstep;%一天是1/360年
X0=X;
%% CIR
obj=cir(Params(1),   @(t,X)Params(2),Params(3),'StartState', 4.2);
[Xcir,Tcxxxx]=obj .simByEuler(Nstep-1,'DeltaTime',1 /360,'nTrials',Nrepl);
%% 生成股票价格路径 每行是一个路径每列是一个时间点
 s=cp*ones(Nrepl,Nstep); %涨停和涨跌规则后的修正股价
%  s=cp*ones(Nrepl,Nstep); 
for j=1:Nrepl
    for i=1:Nstep-1
        r=Xcir(i,j);%CIR
        s(j,i+1)=s(j,i)*exp((r-0.1*(p1-p2)*lamuda-0.5*sigma*sigma)*dt+sigma*sqrt(dt)*randn);%monte carlo模拟股价
    end
end    


% figure
% 
% for i=1:size(s,1)
%    plot(s(i,:));
%    hold on
% end

X=X*ones(Nrepl,1); 
num0=0; 
num1=0; 
p=zeros(Nrepl,1); 
index=zeros(Nrepl,1);
%% 
for j=1:Nrepl%触发回售条款实质是调整股价 
  for k=T-3:T-1    
     for i=(1+k*Nstep/T+round(0.9/(k+1))*0.5*Nstep/T):((k+1)*Nstep/T-30*floor((k+1)/T)) % 转股期限     
         if s(j,i:i+29)<(X(j,1)*0.7) 
            X(j,1)=X(j,1)*mean(s(j,i:i+29));                      
            break        
         end     
     end
     %for a=(T-min(k+1,T-1))*Nstep/T:-1:Nstep/T-30 
     for a=(k*Nstep/T-30):-1:Nstep/T*(T-4)
     if s(j,a:a+19)>=1.3*X(j,1)% 统计转股价修正后触发赎回条款的路径
           p(j,1)=((100/X(j,1))*s(j,a+28)+coupon(1,1:floor(T*a/Nstep))*ones(floor(T*a/Nstep),1))*exp(-r*dt*a); % 理智的投资者会在公司 
           num0=num0+1;
           index(j,1)=1;
               break
        end
     end
  end
end
index2=find(index==0);

if ~isempty(index2)
    for k=length(index2)%统计转股价没有修正相框下触发赎回条款路径
    j=index2(k);
    for i=0.5*Nstep/T:Nstep-30%设置转股期限   
     if (s(j,i:i+19)>=1.3*X0)&(p(j,1)==0)   
         p(j,1)=((100/X0)*s(j,i+28)+coupon(1,1:floor(T*i/Nstep))*ones(floor(T*i/Nstep),1))*exp(-r*dt*i);%理智的投资者会在公司执行赎回条件之前转股。
         num1=num1+1;
         break
     end
    end
end
end
num=num0+num1; 
for m=1:Nrepl%将股票模拟路径中触发赎回去除，剩下的路径转股价格已经确定，完全不受可转债附加条款影响。
 if p(m,1)>0
     s(m,:)=0; 
 end
end
discount=exp(-r*dt); 
discountvet=exp(-r*dt*(1:Nstep)'); 
a=zeros(3,1);
A=100*s(:,Nstep)./X; 
cashflows=max(mcallshedule,A); 
for i=1:Nrepl
    if A(i,1)==0
     cashflows(i,1)=0;
    end 
end
ExerciseTime=Nstep*ones(Nrepl,1); 
matrix_Price=[];
    for step=Nstep-1:-1:1% 从nstep-1步开始，步长为-1，到1结束。
        Inmoney=find(100*s(:,Nstep)./X>100+mean(coupon)*dt*(Nstep-step));%转股有利的。
         y=cashflows(Inmoney).*discountvet(ExerciseTime(Inmoney)-step);%第n步如果转股，在第n-1步贴现。
         St=100*s(Inmoney,step)./X(Inmoney,1);%第n步转股 
         RegrMat=[ones(length(St),1),St,St.^2];%回归矩阵
         a=RegrMat\y;%最小二乘回归
         IntrinsicValue=St;%第n步贴现内在价值
         ContinuationValue=RegrMat*a;%模拟第n+1步贴现的价值。
         Exercise=find(IntrinsicValue>ContinuationValue);%第n步执行转股的状态。
         k=Inmoney(Exercise); 
         Cashflows(k)=IntrinsicValue(Exercise); 
         ExerciseTime(k)=step; 
    end
    Price = mean(cashflows.*discountvet(ExerciseTime)+p(:,1));%平均即为可转债价值。



