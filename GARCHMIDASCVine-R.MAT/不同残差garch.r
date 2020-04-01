
library(tseries)
library(fGarch) 
library(FinTS)
alldata<-read.csv('D:/steam/userdata/299412202/324160/data.csv')

for (iii in 1:480) {
#print(iii)
close<-alldata[,iii]
return<-diff(log(close))
result2<-adf.test(return)$p.value
if (adf.test(return)$p.value<0.05) {
result2<-1
} else {
result2<-0
}

p<-pacf(return)
ubp<-mean(p$acf)+sd(p$acf)*2
lbp<-mean(p$acf)-sd(p$acf)*2

if (p$acf[1]>ubp | p$acf[1]<ubp)  {
result3<-1
} else {
result3<-0
}

if (ArchTest(return)<0.05)  {
print(iii)
m1<-try(garchFit(~arma(1,0)+garch(1,1),data=return,trace=F))
m2<-try(garchFit(~arma(1,0)+garch(1,1),data=return,trace=F),cond.dist = c("std"))
m3<-try(garchFit(~arma(1,0)+garch(1,1),data=return,trace=F),cond.dist = c("ged"))

if ("try-error" %in% class(m1)) &("try-error" %in% class(m2)) & ("try-error" %in% class(m3))  {
longvar<-rep(var(return),2432)
} else {
summary(m1)
longvar<-m1@h.t
}


} else {
longvar<-rep(var(return),2432)
}
if (iii==1)  {
longvarall<-longvar
} else {
longvarall<-rbind(longvarall,longvar)
}
}
write.csv(longvarall,'D:/steam/userdata/299412202/324160/data1.csv')

