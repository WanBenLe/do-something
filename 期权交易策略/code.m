n=5;
f_n=5;
data=xlsread('data.xlsx');
%std
for i=1:(size(data,1)-n+1)
if i==1
s220_v=std(data(i:i+n-1,1));
l220_v=std(data(i:i+n-1,1));
t220_v=std(data(i:i+n-1,1));
else
s220_v=[s220_v,std(data(i:i+n-1,1))];
l220_v=[l220_v,std(data(i:i+n-1,1))];
t220_v=[t220_v,std(data(i:i+n-1,1))];
end
end
m2=mean(s220_v);
std2=std(s220_v);
s220_v=(s220_v-m2)/std2;
m3=mean(l220_v);
std3=std(l220_v);
l220_v=(l220_v-m3)/std3;
s220_v=s220_v';
l220_v=l220_v';
for i=1:(size(s220_v,1)-n+1)
if i==1
s220_x=s220_v(i:i+n-1,1);
l220_x=l220_v(i:i+n-1,1);
else
s220_x=[s220_x,s220_v(i:i+n-1,1)];
l220_x=[l220_x,l220_v(i:i+n-1,1)];
end
end
s220_x=s220_x';
l220_x=l220_x';
%train data
m1=mean(t220_v);
std1=std(t220_v);
t220_v=(t220_v-m1)/std1;
t220_v=t220_v';
for i=1:(size(t220_v,1)-n)
if i==1
t220_xy=t220_v(i:i+n,1);
else
t220_xy=[t220_xy,t220_v(i:i+n,1)];
end
end
t220_xy=t220_xy';
rowrank = randperm(size(t220_xy, 1));
t220_xy= t220_xy(rowrank, :);
% SVR and Optimize
Mdl = fitrsvm(t220_xy(:,1:5),t220_xy(:,6),'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
'expected-improvement-plus'))
plot(t220_xy(:,6));
hold on;
plot(Mdl.resubPredict);
hold off;
legend('Response in Training data','Predicted Response','location','best');
cv = crossval(Mdl);
mse = kfoldLoss(cv)
epsLoss = kfoldLoss(cv,'lossfun','epsiloninsensitive')
%predict and pricing
ys=predict(Mdl,s220_x)*std2+m2;
ys=ys(1:end-1);
yl=predict(Mdl,l220_x)*std3+m3;
yl=yl(1:end-1);
PS220=data(n+f_n-1:end-1,2)*(252*7)^0.5;
PL220=data(n+f_n-1:end-1,3)*(252*7)^0.5;
TP=data(n+f_n-1:end-1,5);
SP=data(n+f_n-1:end-1,1);
RP=data(n+f_n-1:end-1,6);
KP=data(n+f_n-1:end-1,7);
SD1P=log(SP./KP)+(RP+0.5.*PS220.^2).*TP;
SD2P=SD1P-PS220.*TP.^0.5;
SND1 = cdf('Normal',-SD1P,0,1);
SND2 = cdf('Normal',-SD2P,0,1);
ZPP220=KP.*exp(-RP.*TP).*SND2-SP.*SND1;
LD1P=log(SP./KP)+(RP+0.5.*PL220.^2).*TP;
LD2P=LD1P-PL220.*TP.^0.5;
LND1 = cdf('Normal',LD1P,0,1);
LND2 = cdf('Normal',LD2P,0,1);
ZPC220=SP.*LND1-KP.*exp(-RP.*TP).*LND2;
ZP220_t=data(n+f_n:end,2);
ZL220_t=data(n+f_n:end,3);
%

r_1=1.0;
r_2=1.0;
r_3=1.0;

for i=1:(size(ZPP220)-2)
i
if ZPP220(i+1)>ZP220_t(i)
r_1=[r_1,r_1(i)* -((ZP220_t(i+2)-ZP220_t(i+1))/ZP220_t(i+1))+1];

elseif  ZPP220(i+1)<ZP220_t(i)
r_1=[r_1,r_1(i)* ((ZP220_t(i+2)-ZP220_t(i+1))/ZP220_t(i+1)+1)];
else
r_1=[r_1,r_1(i)];
end

if ZPC220(i+1)>ZL220_t(i)
r_2=[r_2,r_2(i)*-((ZL220_t(i+2)-ZL220_t(i+1))/ZL220_t(i+1))+1];
elseif  ZPC220(i+1)<ZL220_t(i)
r_2=[r_2,r_2(i)* ((ZL220_t(i+2)-ZL220_t(i+1))/ZL220_t(i+1)+1)];
else
r_2=[r_2,r_2(i)];
end

if ZPP220(i+1)>ZP220_t(i) & ZPC220(i+1)<=ZL220_t(i)
temp_r=1- ((ZP220_t(i+2)-ZP220_t(i+1))/ZP220_t(i+1)) +((ZL220_t(i+2)-ZL220_t(i+1))/ZL220_t(i+1));
r_3=[r_3,r_3(i)* temp_r];
elseif ZPP220(i+1)<=ZP220_t(i) & ZPC220(i+1)>ZL220_t(i)
temp_r=1+ ((ZP220_t(i+2)-ZP220_t(i+1))/ZP220_t(i+1)) -((ZL220_t(i+2)-ZL220_t(i+1))/ZL220_t(i+1));
r_3=[r_3,r_3(i)* temp_r];
else
r_3=[r_3,r_3(i)];
end
end
plot(r_1);
hold on;
plot(r_2);
plot(r_3);