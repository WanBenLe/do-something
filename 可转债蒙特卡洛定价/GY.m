function Price=GY(cp,X,T,r,coupon,sigma,mcallshedule,Nstep,Nrepl)
%cp:¹ÉÆ±ÏÖ¼Û
%X:Ö´ĞĞ¼Û¸ñ
%T:¿É×ªÕ®Àëµ½ÆÚµÄÊ±¼ä
%r:ÎŞ·çÏÕÀûÂÊ
%coupon:¿É×ªÕ®Ï¢Æ±ÂÊ
%sigma:²¨¶¯ÂÊ
%mcallshcedul:µ½ÆÚÊê»Ø¼Û
%Nstep:Ã¿´ÎÄ£ÄâµÄÆÚÊı
%Nrepl:Ä£ÄâÄÇÃ´¶à´Î
dt=T/Nstep;
s=cp*ones(Nrepl,Nstep);
for j=1:Nrepl
for i=1:Nstep-1
s(j,i+1)=s(j,i)*exp((r-0.5*sigma^2)*dt+sigma*sqrt(dt)*randn);%monte carlo nop
end 
end
X=X*ones(Nrepl,1);
num0=0;
num1=0;
p=zeros(Nrepl,1);
for j=1:Nrepl
for k=0:T-1
for i=floor((1+k*Nstep/T+round(0.9/(k+1))*0.5*Nstep/T)):...
    floor((((k+1)*Nstep/T-30*(k+1)/T)))
if s(j,i:i+29)<(X(j,1)*0.7)
X(j,1)=X(j,1)*mean(s(j,i:i+29));
break
end

end
for a=(T-min(k+1,T-1))*Nstep/T : Nstep/T-30
if s(j,a:a+19)>=1.3*X(j,1)
p(j,1)=((100/X(j,1))*s(j,a+28)+coupon(1,1:floor(T*a/Nstep))*...
ones(floor(T*a/Nstep),1))*exp(-r*dt*a);
num0=num0+1;
break
end
end
end
end
for j=1:Nrepl
for i=floor(0.5*Nstep/T):floor(Nstep-30)
if (s(j,i:i+19)>=1.3*10.14)&(p(j,1)==0)
p(j,1)=((100/10.14)*s(j,i+28)+coupon(1,1:floor(T*i/Nstep))*...
ones(floor(T*i/Nstep),1))*exp(-r*dt*i);%
num1=num1+1;
break
end
end
end
num=num0+num1
for m=1:Nrepl
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
for step=Nstep-1:-1:0.5*Nstep/T
Inmoney=find(100*s(:,Nstep)./X>100+mean(coupon)*dt*(Nstep-step));
y=cashflows(Inmoney).*discountvet(ExerciseTime(Inmoney)-step);
x=100*s(Inmoney,step)./X(Inmoney,1);%?n?,p
RegrMat=[ones(length(x),1),x,x.^2];%??#$
a=RegrMat\y;%^tHu??
IntrinsicValue=x;%?n?%v?
ContinuationValue=RegrMat*a;
Exercise=find(IntrinsicValue>ContinuationValue);
k=Inmoney(Exercise);
Cashflows(k)=IntrinsicValue(Exercise);
ExerciseTime(k)=step;
end
Price=mean(cashflows.*discountvet(ExerciseTime)+p(:,1));