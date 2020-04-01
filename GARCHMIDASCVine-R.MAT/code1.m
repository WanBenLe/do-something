data2=csvread('data2.csv',1,0);
y=data2(1:end,1);
x=data2(1:end,7);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid1=(resid-mean(resid))/std(resid);
Variance1=Variance;
Resid1=resid;

y=data2(1:end,2);
x=data2(1:end,8);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid2=(resid-mean(resid))/std(resid);
Variance2=Variance;
Resid2=resid;

y=data2(1:end,3);
x=data2(1:end,9);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid3=(resid-mean(resid))/std(resid);
Variance3=Variance;
Resid3=resid;

y=data2(1:end,4);
x=data2(1:end,10);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid4=(resid-mean(resid))/std(resid);
Variance4=Variance;
Resid4=resid;

y=data2(1:end,5);
x=data2(1:end,11);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid5=(resid-mean(resid))/std(resid);
Variance5=Variance;
Resid5=resid;


y=data2(1:end,6);
x=data2(1:end,12);
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] =ModifyGarchMidas(y,'X',x);
resid=y-Variance;
resid6=(resid-mean(resid))/std(resid);
Variance6=Variance;
Resid6=resid;

resid_all=[resid1,resid2,resid3,resid4,resid5,resid6];
Resid_all=[Resid1,Resid2,Resid3,Resid4,Resid5,Resid6];
var_all=[Variance1,Variance2,Variance3,Variance4,Variance5,Variance6];
writematrix(Resid_all,'Resid_all.csv');

writematrix(resid_all,'resid_all.csv');
writematrix(var_all,'var_all.csv');