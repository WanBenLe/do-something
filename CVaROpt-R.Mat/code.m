lb = [0,0,0,0,0,0];
ub = [1,1,1,1,1,1];
Aeq = [1,1,1,1,1,1];
beq = 1;
x0=[0.2,0.2,0.1,0.2,0.2,0.1];

x = fgoalattain(@fun,x0,[0 0],[1 100],[],[],[1 1 1 1 1 1],[1],[0 0 0 0 0 0],[]);




disp('------result----')
result=fun(x);
disp(-result(1))
disp(-result(2))
disp('----weight----')
disp(x)