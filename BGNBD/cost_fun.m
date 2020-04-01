function [LL]=cost_fun(Data,one_slove)
% x交易次数
% t时间长度
% T到截止长度
% t_x最后购买时间
% r.alpha.a.b

row_num=size(Data.data0,1);

r=one_slove(1);
alpha=one_slove(2);
a=one_slove(3);
b=one_slove(4);

LL=0.0;
for i=1:row_num
    x=Data.data0(i,1);
    T=Data.data0(i,2);
    t_x=Data.data0(i,3);
    A1_temp=gammaln(r+x)-gammaln(r)+r*log(alpha);
    A.A1_temp=A1_temp;
    A2_temp=gammaln(a+b)+gammaln(b+x)-gammaln(b)-gammaln(a+b+x);
    A.A2_temp=A2_temp;
    A3_temp=-(r+x)*log(alpha+T);
    A.A3_temp=A3_temp;
    A4_temp=log(a)-log(b+x-1)-(r+x)*log(alpha+t_x);
    if x<=0.0
        A4_temp=0.0;
        A.A4_temp=A4_temp;
    end  
    if x>0.0
        LL=LL+A1_temp+A2_temp+log(exp(A3_temp)+exp(A4_temp));
    else
        LL=LL+A1_temp+A2_temp+log(exp(A3_temp));
    end
end


