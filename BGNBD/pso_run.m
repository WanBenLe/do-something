
% 种群数量.迭代次数.惯性权重.局部学习因子.全局学习因子.信息传输概率.最低权重
% 增加这两个会增加计算时间,也增加获得最优解的可能性
% 尽量不要使得num_k跟num_n差不多(此行可删)
num_n=20;
num_k=100;

% 就结论而言如果不懂这是什么,不要随便修改(此行可删)
par_1=0.7;
par_2=2.05;
par_3=2.05;
par_4=0.66;
par_5=0.3;

hello=4;

% 计算每次迭代下降的惯性权重
w_down=(par_1-par_5)/num_k; 

% 读取数据
[Data.data0,~] = xlsread('data0.xlsx');


            
% PSO算法
% 0个参数为0是初始解
g_best=rand(4,1)*hello
g_cost=cost_fun(Data,g_best)
cost_num=[g_cost];
g_cost1=g_cost;

for i=1:num_k
	for j=1:num_n
		% 生成随机解
		one_slove=rand(4,1)*hello;
		one_cost=cost_fun(Data,one_slove);
		% 局部最优和全局最优的更新和初始化
        if one_cost>g_cost   && (one_slove(3))>0.1 && (one_slove(4))<4
			p_best=one_slove;
			g_best=one_slove;
			p_cost=one_cost;
			g_cost=one_cost;
		elseif j==1|| (one_cost>p_cost && one_slove(3)>0.1 && one_slove(4)<4 
			p_best=one_slove;
			p_cost=one_cost;
		% 受到p_best和g_best影响,更新cost
		elseif rand(1,1)<par_4
            for k=1:4
				one_slove(k)=one_slove(k)*par_1+rand(1,1)*par_2*(p_best(k)-one_slove(k))+rand(1,1)*par_3*(g_best(k)-one_slove(k));
                if one_slove(k)<0.0001
                    one_slove(k)=0.0001;
                end
            end
            
			one_cost=cost_fun(Data,one_slove);
			% 更新g_best和p_best
			if one_cost>p_cost  && (one_slove(3))>0.1 && (one_slove(4))<4 
				p_best=one_slove;
				p_cost=one_cost;
			end
			if one_cost>g_cost  && (one_slove(3))>0.1 && (one_slove(4))<4 
				g_best=one_slove;
				g_cost=one_cost;  
			end
        end
	end
	% 下降权重
    cost_num=[cost_num;g_cost];
	par_1=par_1-w_down;
end

% 作图
x=1:1:size(cost_num,1);  
plot(x,cost_num,'-r');  
hold on  
polt_1=-1150;
polt_2=-950;
axis([1,size(cost_num,1),polt_1,polt_2])  
set(gca,'XTick',[0:num_k/10:size(cost_num,1)]) 

set(gca,'YTick',[polt_1:0.2*(polt_2-polt_1):polt_2]) 
xlabel('最优LL变化曲线')  
ylabel('LL')  


'r.alpha.a.b为'
g_best'

'极大似然函数为'
LL=cost_fun(Data,g_best)

