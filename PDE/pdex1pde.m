function [c,f,s] = pdex1pde(x,t,u,DuDx)
c = 1;
f = (1.985*10^-7)*DuDx;
s = 0;