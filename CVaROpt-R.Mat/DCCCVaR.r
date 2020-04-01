
##################
#3.6.1以上的R,先运行这个

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")


#用这三条命令安装包XXX
BiocManager::install()
options(repos='http://cran.rstudio.com/')
BiocManager::install("XXX")


#################

library(quantreg)
library(ccgarch)
library(fGarch)
library(MASS)
library(FinTS)
library(tseries)
library(vars)
library(forecast)
library(rugarch)


#读取数据
data<-read.csv('C:/Users/data.csv')




#ADFtest
adf.test(data[,2])
#差分
if (adf.test(data[,2])$p.value<0.05){
data2<-diff(data[,2])
}else{data2<-data[2:length(data[,2]),2]}
adf.test(data2)
#ARIMA
a1<-auto.arima(data2,trace = TRUE)
#残差标准化
y<-scale(a1$residuals, center = TRUE, scale = TRUE)

adf.test(data[,3])
if (adf.test(data[,3])$p.value<0.05){
data3<-diff(data[,3])
}else{data3<-data[2:length(data[,3]),2]}
adf.test(data3)
a1<-auto.arima(data3,trace = TRUE)
z<-scale(a1$residuals, center = TRUE, scale = TRUE)



#ARCH检验,或许要去掉
ArchTest(y,lags=3,demean=T)
ArchTest(z,lags=3,demean=T)
#GARCH
garch.y=garch(y,order=c(1,1))
summary(garch.y)
garch.z=garch(z,order=c(1,1))
summary(garch.z)
#大概写一下初始参数
inia=c(3.266e-07 , 1.436e-07)
iniA=diag(c( 5.791e-02  ,5.125e-02))
iniB=diag(c(9.403e-01, 9.469e-01))
nobs=length(y)
dcc.para<-c(0.01,0.97)
inicor=cor(y,z)
uncR=matrix(c(1,inicor,inicor,1),2,2)
dcc.results <- dcc.estimation(inia, iniA, iniB, dcc.para, dvar=cbind(y,z), model="diagonal")
#系数,动态相关系数,作图等
out<-dcc.results$out
out
DCC<-dcc.results$DCC[,2]
DCC
ts.plot(DCC)
#分位数回归
fit1 = rq(data[,2] ~ data[,3], tau = 0.1)
summary(fit1, se = 'nid') 
fit1 = rq(data[,2] ~ data[,3], tau = 0.5)
summary(fit1, se = 'nid') 
fit1 = rq(data[,2] ~ data[,3], tau = 0.9)
summary(fit1, se = 'nid') 