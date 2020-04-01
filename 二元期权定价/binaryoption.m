%Moeti Ncube: This code can be used to price binary options. A binary
%options have a payoff of 0 or 1. I wrote this code to price the fair value
%of the Intrade.com contract: (DOW to close HIGHER than prev close).

%Current Dow price
S=12017;
%Previous Dow Close
K=11961.52;
%Risk Free Rate
r=0.03;
%Dividend Yield
q=0;
%DJIA volatility (VXD Index)
sigma=.1910;
%Time between now and close of Dow (4:00PM)
time1=datevec(now);
time2=[time1(1:3),16,0,0];
T=(datenum(time2)-datenum(time1))/365;



d1=(log(S/K)-(r-q+.5*sigma^2)*T)/(sigma*sqrt(T));
d2=d1-sigma*sqrt(T);

%Call Option price
Call=exp(-r*T)*normcdf(d2,0,1)
%Put Option Price
Put=exp(-r*T)*normcdf(-d2,0,1)
