yszj=xlsread('基础资产现金流预算表.xlsx');
dt=1/360;
nTrials=100;
strm=RandStream('mt19937ar','seed',14151617);
RandStream.setGlobalStream(strm);
times=yszj(:,1);
yszj=yszj(:,2:end);
obj=cir(9.7252,   @(t,X)2.5395,2.1313,'StartState', 2.799);
[X,T]=obj .simByEuler(1500,'DeltaTime',1 /360,'nTrials',nTrials);
DCF=zeros(30,nTrials);
SumVal=zeros(1,nTrials);
Dcf = zeros(1,30);
Aa=30;
for k=1:nTrials
    for j=1:Aa
        DCF(j,k)=yszj(j)*exp(-X(30*j,k)*0.01);
    end
    Val=sum(DCF(:,k));
    B=SumVal(1,k);
    SumVal( 1 ,k)=B+Val;
end
DCF(31,:)=SumVal;
csvwrite('dcf1000.csv',DCF)
xxx=[];
for i=1:31
xxx(i)=mean(DCF(i,:));
end
xxx=xxx';
xxx(end)
'finished'