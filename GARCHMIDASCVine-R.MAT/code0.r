install.packages("VineCopula")
install.packages("tseries")
install.packages("copula")
install.packages("rugarch")
install.packages("FinTS")
install.packages("vars")
install.packages("forecast")

library(VineCopula)
library(tseries)
require(copula)
require(rugarch)
library(FinTS)
library(vars)
library(forecast)

datax<-read.csv('C:/Users/70307/Desktop/zzzzz/data1.csv')
data1<-diff(datax[,1])
data2<-diff(datax[,2])
data3<-diff(datax[,3])
data4<-diff(datax[,4])
data5<-diff(datax[,5])
data6<-diff(datax[,6])

rm(data)
data<-cbind(data1,data2,data3,data4,data5,data6)
write.csv(data,'C:/Users/70307/Desktop/zzzzz/data.csv')

if(Box.test(data[,1])$p.value<0.05){
a1<-auto.arima(data[,1],trace = TRUE)
hbrys1<-a1$residuals
}else{hbrys1<-data[,1]}
Box.test(hbrys1)


if(Box.test(data[,2])$p.value<0.05){
a1<-auto.arima(data[,2],trace = TRUE)
hbrys2<-a1$residuals
}else{hbrys2<-data[,2]}
Box.test(hbrys2)


if(Box.test(data[,3])$p.value<0.05){
a1<-auto.arima(data[,3],trace = TRUE)
hbrys3<-a1$residuals
}else{hbrys3<-data[,3]}
Box.test(hbrys3)


if(Box.test(data[,4])$p.value<0.05){
a1<-auto.arima(data[,4],trace = TRUE)
hbrys4<-a1$residuals
}else{hbrys4<-data[,4]}
Box.test(hbrys4)


if(Box.test(data[,5])$p.value<0.05){
a1<-auto.arima(data[,5],trace = TRUE)
hbrys5<-a1$residuals
}else{hbrys5<-data[,5]}
Box.test(hbrys5)


if(Box.test(data[,6])$p.value<0.05){
a1<-auto.arima(data[,6],trace = TRUE)
hbrys6<-a1$residuals
}else{hbrys6<-data[,6]}
Box.test(hbrys6)


rm(data)
data<-cbind(hbrys1,hbrys2,hbrys3,hbrys4,hbrys5,hbrys6)
write.csv(data,'C:/Users/70307/Desktop/zzzzz/resid.csv')

#C��
data<-pobs(data)
CRVM <- RVineStructureSelect(data, c(1:6), "CVine")
simdata <-data
'MLE'
mle <- RVineMLE(data, CRVM, grad = TRUE, trace = 0)
round(mle$CRVM$par - CRVM$par, 2)
'AIC'
RVineAIC(data, CRVM)
'BIC'
RVineBIC(data, CRVM)
'tau'
tau <- RVinePar2Tau(CRVM)
summary(CRVM)
'polt'
#作图
contour(CRVM)
plot(CRVM,tree=1,type=1,edge.labels="family-tau",interactive = TRUE)
#得到模拟�?
A<-RVineSim(1788,RVM, U = NULL)
write.csv(A,'D:/Desktop/sim1C.csv')
B<-pnorm(A)
temp<-c(1:101)

#R��
#选择最优Vine Copula
#RVM是最优结�?
RVM <- RVineStructureSelect(data, c(1:6), "RVine")
simdata <- data
'MLE'
mle <- RVineMLE(data, RVM, grad = TRUE, trace = 0)
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
plot(RVM,tree=1,type=1,edge.labels="family-tau",interactive = TRUE)
#得到模拟�?
A<-RVineSim(1788,RVM, U = NULL)
write.csv(A,'D:/Desktop/sim1R.csv')
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
write.csv(cool,'D:/Desktop/COOLR.csv')
