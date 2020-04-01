library(VineCopula)
library(tseries)
require(copula)
require(rugarch)
library(FinTS)
library(vars)
library(forecast)

data<-read.csv('C:/Users/70307/Desktop/zzzzz/resid_all.csv')

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
A<-RVineSim(1788,CRVM, U = NULL)
write.csv(A,'C:/Users/70307/Desktop/zzzzz/sim1C.csv')
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
write.csv(A,'C:/Users/70307/Desktop/zzzzz/sim1R.csv')
B<-pnorm(A)
temp<-c(1:101)


datax<-read.csv('C:/Users/70307/Desktop/zzzzz/data1.csv')
modelres<-read.csv('C:/Users/70307/Desktop/zzzzz/Resid_all.csv')
#--------------
cool1<-datax[-1,1]+quantile(modelres[,1],(B[,1]))-modelres[,1]
cool2<-datax[-1,2]+quantile(modelres[,2],(B[,2]))-modelres[,2]
cool3<-datax[-1,3]+quantile(modelres[,3],(B[,3]))-modelres[,3]
cool4<-datax[-1,4]+quantile(modelres[,4],(B[,4]))-modelres[,4]
cool5<-datax[-1,5]+quantile(modelres[,5],(B[,5]))-modelres[,5]
cool6<-datax[-1,6]+quantile(modelres[,6],(B[,6]))-modelres[,6]
cool<-cbind(cool1,cool2,cool3,cool4,cool5,cool6)
write.csv(cool,'C:/Users/70307/Desktop/zzzzz/COOLR.csv')
