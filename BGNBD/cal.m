r=2.26;
alpha=3.6712;
a=0.1182;
b=3.5648;
T=60;

[data,~] = xlsread('data0.xlsx');
row=size(data);
exp=[];
for i=1:row
    x_1=data(i,1);
    t=data(i,2);
    t_x=data(i,3);
    p1=(a+b+x_1-1)/(a-1);
    p2=hypergeom([r+x_1 b+x_1],b+x_1-1 ,(t/(alpha+T+t))) ;
    p3=1-(((alpha+T)/(alpha+T+t))^(r+x_1)*p2);
    p4=1+ (alpha/(b+x_1-1)) *  ((alpha+T)/(alpha+t_x))^(r+x_1);
    exp=[exp;p1*p3/p4];
end
exp