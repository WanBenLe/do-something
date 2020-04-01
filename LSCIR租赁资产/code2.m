data=xlsread('基础资产现金流预算表.xlsx');
dt=1/360;
nTrials=1000;
strm=RandStream('mt19937ar','Seed',14151617);
RandStream.setGlobalStream(strm);
obj=cir(9.7252,   @(t,X)2.5395,2.1313,'StartState', 2.799);
[X, T]=obj.simByEuler(1100,'DeltaTime',1/360,'nTrials',1000);
DCF=zeros(10,40,1000);
SumVal=0*ones(nTrials);
Dcf=zeros(10,40);
for i=1:10
Aa=data(:,1);
Ba = data(:,2);
Ca=data(:,3);
for k=1:nTrials
for j=1:(Aa-1)
times =T(30*j);
CRPY=0.06*j/30;
DEFY=0.06*j/30;
SMS=1-(1-CRPY)^(1/12);
DMS=1-(1-DEFY)^(1/12);
CFS(i,j,k)=Ba*(1-DMS) *Ca/12+Ba*SMS;
Ba=Ba*(1-DMS-SMS);
DCF(i,j,k)=CFS(i,j,k) *exp(-times*X(30*j,k)*0.01);
j=j+1;
DCF(i,j,k)=(Ba*(1-DMS)*Ca/12+Ba*SMS)*exp(-times*X(30*j,k)*0.01)+Ba*(1-DMS)*exp(-times*X(30*Aa,1)*0.01);
disp(DCF)
end
Val(j,j)=sum(DCF(i,:,k));
B=SumVal(j,j);
SumVal(j,j)=B+Val(j,j);
Ba=data(i,2);
end
for p=1:Aa
M(i,p)=mean(DCF(i,p,:));
end
AvgVal= SumVal(j,j)/nTrials;
disp(AvgVal)
end
for j=1:10
    DisCF(j)=sum(M(j,:));
end
