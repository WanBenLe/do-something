library(VineCopula)
library(tseries)
require(copula)
require(rugarch)
library(FinTS)
library(vars)
library(forecast)

datax<-read.csv('C:/Users/70307/Desktop/cool/data1.csv')
data1<-diff(datax[,1])
data2<-diff(datax[,2])
data3<-diff(datax[,3])
data4<-diff(datax[,4])
data5<-diff(datax[,5])
data6<-diff(datax[,6])

rm(data)
data<-cbind(data1,data2,data3,data4,data5,data6)


if(Box.test(data[,1])$p.value<0.05){
a1<-auto.arima(data[,1],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,1]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit1<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit1)
res1<-as.numeric(residuals(myfit1,standardize=TRUE))
}else{res1<-hbrys}

if(Box.test(data[,2])$p.value<0.05){
a1<-auto.arima(data[,2],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,2]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit2<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit2)
res2<-as.numeric(residuals(myfit2,standardize=TRUE))
}else{res2<-hbrys}

if(Box.test(data[,3])$p.value<0.05){
a1<-auto.arima(data[,3],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,3]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit3<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit3)
res3<-as.numeric(residuals(myfit3,standardize=TRUE))
}else{res3<-hbrys}

if(Box.test(data[,4])$p.value<0.05){
a1<-auto.arima(data[,4],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,4]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit4<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit4)
res4<-as.numeric(residuals(myfit4,standardize=TRUE))
}else{res4<-hbrys}

if(Box.test(data[,5])$p.value<0.05){
a1<-auto.arima(data[,5],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,5]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit5<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit5)
res5<-as.numeric(residuals(myfit5,standardize=TRUE))
}else{res5<-hbrys}

if(Box.test(data[,6])$p.value<0.05){
a1<-auto.arima(data[,6],trace = TRUE)
hbrys<-a1$residuals
}else{hbrys<-data[,6]}
Box.test(hbrys)

ArchTest(hbrys,lags=15)$p.value
if (ArchTest(hbrys,lags=15)$p.value<0.05){
myspec=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "norm"
)
myfit6<-ugarchfit(myspec,data=hbrys,solver="gosolnp")
print(myfit6)
res6<-as.numeric(residuals(myfit6,standardize=TRUE))
}else{res6<-hbrys}

rm(data)
data<-cbind(res1,res2,res3,res4,res5,res6)

#选择最优Vine Copula
#RVM是最优结构
data<-pobs(data)
RVM <- RVineStructureSelect(data, c(1:6), "RVine")
simdata <- RVineSim(300, RVM)
'MLE'
mle <- RVineMLE(simdata, RVM, grad = TRUE, trace = 0)
round(mle$RVM$par - RVM$par, 2)
'AIC'
RVineAIC(simdata, RVM)
'BIC'
RVineBIC(simdata, RVM)
'tau'
tau <- RVinePar2Tau(RVM)
summary(RVM)
'polt'
#作图
contour(RVM)
#得到模拟值
A<-RVineSim(1788,RVM, U = NULL)
#write.csv(A,'C:/Users/70307/Desktop/cool/sim.csv')
B<-pnorm(A)
temp<-c(1:101)

#--------------
cool1<-datax[-1,1]+quantile(residuals(myfit1),(B[,1]))-(residuals(myfit1))
cool2<-datax[-1,2]+quantile(residuals(myfit2),(B[,2]))-(residuals(myfit2))
cool3<-datax[-1,3]+quantile(residuals(myfit3),(B[,3]))-(residuals(myfit3))
cool4<-datax[-1,4]+quantile(residuals(myfit4),(B[,4]))-(residuals(myfit4))
cool5<-datax[-1,5]+quantile(residuals(myfit5),(B[,5]))-(residuals(myfit5))
cool6<-datax[-1,6]+quantile(residuals(myfit6),(B[,6]))-(residuals(myfit6))
cool<-cbind(cool1,cool2,cool3,cool4,cool5,cool6)
write.csv(cool,C:/Users/70307/Desktop/cool/COOLR.csv')
