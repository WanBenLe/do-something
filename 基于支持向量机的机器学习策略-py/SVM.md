# 支持向量机

## 支持向量机简介

### 什么是支持向量机？

​	支持向量机(Support Vector Machines, SVM) 被Vapnik与他的合作者提出于1995年,基础为统计学习理论和结构风险最小化原则.支持向量机具有完备的理论基础和出色的学习能力,是借助于最优化方法解决有限样本机器学习问题的数据挖掘出色方法之一.

### 支持向量机的原理是？

​	假设给定一个特征空间上的训练集:

$$T=\begin{Bmatrix}(x_{1},y_{1}),(x_{2},y_{2}),...,(x_{i},y_{i})\end{Bmatrix};x_{i}\in \boldsymbol{X}=\mathbb{R}^n,y_{i}\in \boldsymbol{Y}=\begin{Bmatrix}+1,-1\end{Bmatrix};i=1,2,...,N$$

​	其中$x_{i}$为第$i$个特征向量,$y_{i}$为$x_{i}$的类标记,当$y_{i}=+1$时,$x_{i}$称为正例,当$y_{i}=-1$时,$x_{i}$称为负例.

#### 线性可分SVM

​	假设训练数据集线性$T$可分,通过间隔最大化或等价求解相应凸二次规划问题得到分离超平面$w^{*}\cdot x+b^{*}=0$和相应的分类决策函数$f(x)=sign(w^{*}\cdot x+b^{*})$为线性可分SVM.

​	为了量化分类的正确性和确信度,引入函数间隔的概念.

​	对于给定的训练数据集$T$和超平面$(w,b)$,定义超平面关于样本点$(x_{i},y_{i})$的函数间隔为:

<center>$$ \hat{\gamma_{i}}=y_{i}(w\cdot x_{i}+b)$$</center>

​	定义超平面$(w,b)$关于训练数据集$T$的函数间隔为关于$T$中所有样本点$(x_{i},y_{i})$的函数间隔的最小值:

<center>$\hat{\gamma}=\min_{i=1,...,N}\hat{\gamma_{i}}$</center>

​	为了取消成比例改变$w.b$导致函数间隔变化但超平面不变的问题,引入规范化$\left \| w \right \|$,此时函数间隔变为几何间隔.

​	对于给定的训练数据集$T$和超平面$(w,b)$,定义超平面关于样本点$(x_{i},y_{i})$的几何间隔为:

<center>$$ \hat{\gamma_{i}}=y_{i}(\frac{w}{\left \| w \right \|}\cdot x_{i}+\frac{b}{\left \| w \right \|})$$</center>

​	定义超平面$(w,b)$关于训练数据集$T$的函数间隔为关于$T$中所有样本点$(x_{i},y_{i})$的几何间隔的最小值:

<center>$\hat{\gamma}=\min_{i=1,...,N}\hat{\gamma_{i}}$</center>

​	可知函数间隔和几何间隔的关系为:

<center>$\gamma_{i}=\frac{\hat{\gamma_{i}}}{\left \| w \right \|}$,$\gamma=\frac{\hat{\gamma}}{\left \| w \right \|}$	$(1)$</center>												

##### 硬间隔最大化

​	通过寻找最大化几何间隔的分离超平面可以以充分大的确信度对训练数据进行分类,最大化几何间隔又称为硬间隔最大化,此时的约束最优化问题为:

<center>$$\max_{w,b} \gamma$$</center>

<center>$$s.t. y_{i}(\frac{w}{\left \| w \right \|}\cdot x_{i}+\frac{b}{\left \| w \right \|})s\geqslant \gamma ,i=1,2,...,N$$</center>

​	即最大化超平面$(w,b)$关于训练数据集$T$的几何间隔$\gamma$

​	考虑$(1)$,问题可改写为:

<center>$$\max_{w,b} \frac{\hat \gamma}{\left \| w \right \|}$$</center>

<center>$$s.t. y_{i}(w\cdot x_{i}+b)\geqslant  \hat\gamma ,i=1,2,...,N$$</center>

​	基于与引入规范化$\left \| w \right \|$同样的原因,$\hat \gamma$的取值对结果没有影响,故取$\hat \gamma=1$,考虑

<center>$$\max_{w,b} \frac{1}{\left \| w \right \|}\Leftrightarrow \min_{w,b}\frac{1}{2}\left \| w \right \|^{2}$$</center>

​	有以下线性可分SVM学习的最优化问题:

<center>$$\min_{w,b}\frac{1}{2}\left \| w \right \|^{2}$$	$(2)$</center>													

<center>$$s.t. y_{i}(w\cdot x_{i}+b)-1\geqslant  0,i=1,2,...,N$$	$(2)$</center>							

​	该问题是一个凸二次规划问题.可以证明,若训练数据集$T$线性可分,那么可将训练数据集中的样本点完全正确分开的最大间隔超平面存在且唯一.

​	在线性可分的情况下,训练数据集的样本点与分离超平面距离最近的样本点的示例称为支持向量,如下图的$H_{1}$和$H_{2}$上的点:

![支持向量](SV.jpg)

​	$H_{1}$和$H_{2}$之间的距离称为间隔,为$\frac{2}{\left \| w \right \|}$,$H_{1}$和$H_{2}$称为间隔边界.

##### 线性可分SVM对偶学习算法

​	通过运用拉格朗日对偶性,可以得到原始问题的对偶问题,对偶问题往往更容易求解,也便于引入核函数推广到非线性分类问题.拉格朗日对偶性在此不赘述,但有个重要定理需要特别指出:对于原始问题和对偶问题,在满足特定条件时,则$x^{*}.\alpha^{*}. \beta ^{*}$分别为原始问题和对偶问题的解的充要条件为$x^{*}.\alpha^{*}. \beta ^{*}$满足KKT条件.KKT条件表述如下:

<center>$$\bigtriangledown _{x}L(x^{*},\alpha^{*}, \beta ^{*})=0$$</center>

<center>$$\alpha_{i}^{*}c_{i}(x^{*})=0,i=1,2,...,k$$(KKT的对偶互补条件)</center>

<center>$$c_{i}(x^{*})\leqslant 0,i=1,2,...,k$$</center>

<center>$$\alpha_{i}^{*}\geqslant 0,i=1,2,...,k$$</center>

<center>$$h_{j}(x^{*})=0,j=1,2,...,l$$</center>

​	向$(2)$引入拉格朗日乘子$\alpha_{i}\geq 0,i=1,2,...,N$定义拉格朗日函数

<center>$$L(w,b,\alpha)=\frac{1}{2}\left \| w \right \|^{2}-\sum_{i=1}^{N}\alpha_{i}y_{i}(w\cdot x_{i}+b)+\sum_{i=1}^{N}\alpha_{i}$$	$(3)$</center>			

​	其中,$\alpha=(\alpha_{1},\alpha_{2},...,\alpha_{N})^{T}$为拉格朗日乘子向量.

​	根据拉格朗日对偶性可以得到原始问题的对偶问题即极大极小问题:

<center>$$\max_{\alpha}\min_{w,b}L(w,b,\alpha)$$</center>

​	先求$\min_{w,b}L(w,b,\alpha)$,分别让$(3)$对$w.b$的偏导等于0,有:

<center>$$\bigtriangledown _{w}L(w,b, \alpha)=w-\sum_{i=1}^{N}\alpha_{i}x_{i}y_{i}=0$$</center>

<center>$$\bigtriangledown _{b}L(w,b, \alpha)=-\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$</center>

<center>$$w=\sum_{i=1}^{N}\alpha_{i}x_{i}y_{i}$$	$(4)$</center>													

<center>$$\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$	$(4)$</center>													

​	把$(4)$代入$(3)$整理可得:

<center>$$\min_{w,b}L(w,b,\alpha)=-\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})+\sum_{i=1}^{N}\alpha_{i}$$</center>

​	接下来再求$\max_{\alpha}\min_{w,b}L(w,b,\alpha)$,其对偶问题为:

<center>$$\max_{\alpha}-\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})+\sum_{i=1}^{N}\alpha_{i}$$	$(5)$</center>					

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$	$(5)$</center>											

<center>$$\alpha_{i}\geqslant 0,i=1,2,...,N$$	$(5)$</center>											

​	$(5)$等价于下面的最优化问题:

<center>$$\min_{\alpha}\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})-\sum_{i=1}^{N}\alpha_{i}$$	$(6)$</center>						

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$	$(6)$</center>												

<center>$$\alpha_{i}\geqslant 0,i=1,2,...,N$$	$(6)$</center>																								

​	可以证明以下定理:设$\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},...,\alpha_{i}^{*})^{T}$为对偶优化问题$(6)$的解,则$\exists j,\alpha_{j}^{*}> 0$且原始最优化问题可按下式求解$w^{*}.b^{*}$:

<center>$$w^{*}=\sum_{i=1}^{N}\alpha_{i}^{*}x_{i}y_{i}	$$</center>

<center>$$b^{*}=y_{j}-\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x_{i}\cdot x_{j})$$</center>

​	此时有分离超平面:

<center>$$\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x\cdot x_{i})+b^{*}=0$$</center>

​	分类决策函数:

<center>$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x\cdot x_{i})+b^{*}]$$</center>

​	考虑原始最优化问题$(2)$和对偶最优化问题$(6)$,将数据集中对应$\alpha_{i}^{*}>0$的样本点$(x_{i},y_{i})$的实例$x_{i}\in \mathbb{R}^{n}$称为支持向量.可证明支持向量$x_{i}$一定在间隔边界上.

##### 线性可分SVM学习算法

构造求解约束最优化问题

<center>$$\min_{\alpha}\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})-\sum_{i=1}^{N}\alpha_{i}$$</center>			

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$</center>						

<center>$$\alpha_{i}\geqslant 0,i=1,2,...,N$$</center>	

求得最优解$\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},...,\alpha_{i}^{*})^{T}$

计算$w^{*}=\sum_{i=1}^{N}\alpha_{i}^{*}x_{i}y_{i}$并选取$\alpha_{j}^{*}>0$计算$b^{*}=y_{j}-\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x_{i}\cdot x_{j})$.

求得分离超平面:$w^{*}\cdot x+b^{*}=0$

分类决策函数:$f(x)=sign(w^{*}\cdot x+b^{*})$

#### 软间隔最大化线性SVM

​	假设训练数据集$T$线性不可分且存在特异点,除特异点以外的数据线性可分.

​	为了使得线性不可分而不满足$(2)$中约束条件的的训练数据集$T$可被训练,对每个样本点$(x_{i},y_{i})$引入松弛变量$\xi \geq 0$,并支付相应的距离代价$\xi_{i} $,使得$(2)$变为:

<center>$$\min_{w,b}\frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{N}\xi_{i}$$</center>						

<center>$$s.t. y_{i}(w\cdot x_{i}+b)\geqslant  1-\xi_{i},i=1,2,...,N$$</center>

​	其中$C>0$,称为惩罚系数,取值由实际问题决定.这个思路称为软间隔最大化.

​	此时线性不可分SVM的学习问题变为如下的凸二次规划原始问题:

<center>$$\min_{w,b}\frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{N}\xi_{i}$$	$(7)$</center>													

<center>$$s.t. y_{i}(w\cdot x_{i}+b)\geqslant  1-\xi_{i},i=1,2,...,N$$	$(7)$</center> 							

<center>$$\xi_{i}>0,i=1,2,...,N$$	$(7)$</center>										

​	可以证明$\exists (w,b,\xi)$为$(7)$的解,且$w$唯一,$b$可能不唯一且存在于一个区间中.

​	$(7)$的对偶问题是:

<center>$$\min_{\alpha}\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})-\sum_{i=1}^{N}\alpha_{i}$$	$(8)$</center>						

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$	$(8)$</center>													

<center>$$0\leq \alpha_{i}\leq C,i=1,2,...,N$$	$(8)$</center>									

##### 线性SVM学习算法

选择惩罚参数$C>0$,构造并求解凸二次规划问题

<center>$$\min_{\alpha}\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})-\sum_{i=1}^{N}\alpha_{i}$$</center>			

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$</center>						

<center>$$C\geqslant \alpha_{i}\geqslant 0,i=1,2,...,N$$</center>	

求得最优解$\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},...,\alpha_{i}^{*})^{T}$

计算$w^{*}=\sum_{i=1}^{N}\alpha_{i}^{*}x_{i}y_{i}$并选取$C>\alpha_{j}^{*}>0$计算$b^{*}=y_{j}-\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x_{i}\cdot x_{j})$.

求得分离超平面:$w^{*}\cdot x+b^{*}=0$

分类决策函数:$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x\cdot x_{i})+b^{*}]$$

#### 非线性SVM

​	假设训练数据集$T$能用$\mathbb{R}^{n}$中的一个超曲面将正负例分开,这种分类问题被称为非线性可分问题.

​	对于这种问题,需要应用SVM的核技巧如下:

​	通过非线性变换将输入空间(欧式空间$\mathbb{R}^{n}$或离散集合)对应于一个特征空间(希尔伯特空间$\mathbb{H}$)

​	输入空间$\mathbb{R}^{n}$中的超曲面模型对应于特征空间$\mathbb{H}$中的超平面模型(SVM)

​	通过求解特征空间$\mathbb{H}$中的线性SVM完成非线性可分分类问题的学习任务.

​	设$\boldsymbol{X}$为输入空间(欧式空间$\mathbb{R}^{n}$的子集或离散集合),又称$\mathbb{H}$为特征空间(希尔伯特空间),若$\exists \phi (x):\boldsymbol{X}\rightarrow \mathbb{H}$使得$\forall x.z\in \boldsymbol{X},K(x,z)=\phi (x)\cdot \phi (z)$,则称$K(x,z)$为核函数,$\phi (x)$称为映射函数.$\phi (x)\cdot \phi (z)$称为$\phi(x)$和$\phi(z)$的内积.

​	用核函数$K(x,z)​$代替对偶问题$(8)​$中的目标函数和分类决策函数的内积,有新的目标函数:

<center>$$W(\alpha) = \frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}K(x_{i},x_{j})-\sum_{i=1}^{N}\alpha_{i}$$</center>

​	和新的分类决策函数:

<center>$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}K(x_{i},x)+b^{*}]$$</center>

​	若使用多项式核函数$K(x,z)=(x \cdot z+1 )^{p}$,对应的SVM为$p$次多项式分类器,分类决策函数成为:

<center>$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}(x_{i}\cdot x+1)^{p}+b^{*}]$$</center>

​	若使用高斯核函数$K(x,z)=exp(-\frac{\left \| x-z \right \|^{2}}{2\sigma ^{2}})$,对应的SVM为高斯径向基函数(RBF)分类器,分类决策函数成为:

<center>$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}exp(-\frac{\left \| x-z \right \|^{2}}{2\sigma ^{2}})+b^{*}]$$</center>

##### 非线性SVM学习算法

选取适当核函数$K(x,z)$和适当的惩罚参数$C>0$,构造并求解最优化问题

<center>$$\min_{\alpha}\frac{1}{2}\sum_{j=1}^{N}\sum_{i=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}K(x_{i},x_{j})-\sum_{i=1}^{N}\alpha_{i}$$	$(9)$</center>						

<center>$$s.t.\sum_{i=1}^{N}\alpha_{i}y_{i}=0$$	$(9)$</center>													

<center>$$C\geqslant \alpha_{i}\geqslant 0,i=1,2,...,N$$	$(9)$</center>									

求得最优解$\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},...,\alpha_{i}^{*})^{T}$

并选取$C>\alpha_{j}^{*}>0$计算$b^{*}=y_{j}-\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}K(x_{i},x_{j})$.

构造分类决策函数:$$f(x)=sign[\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}K(x,x_{i})+b^{*}]$$

可以证明,当$K(x,z)$为正定核函数时,$(9)$为凸二次规划问题并有解.

## 附录

### B. Gram矩阵

定义:$n$维欧式空间任意$k(k\leq n)$个向量$\alpha_{1},\alpha_{2},...,\alpha_{k}$的内积组成的矩阵

<center>$\Delta (\alpha_{1},\alpha_{2},...,\alpha_{k})=\begin{vmatrix} (\alpha_{1},\alpha_{1}) &(\alpha_{1},\alpha_{2})  & ... & (\alpha_{1},\alpha_{k})\\ (\alpha_{2},\alpha_{1}) &(\alpha_{2},\alpha_{2})  & ... & (\alpha_{2},\alpha_{k}) \\...  & ...  & ... & ... \\ (\alpha_{k},\alpha_{1}) &(\alpha_{k},\alpha_{2})  & ... & (\alpha_{k},\alpha_{k}) \end{vmatrix}$</center>

​	称为$k$个向量$\alpha_{1},\alpha_{2},...,\alpha_{k}$的Gram矩阵,行列式$G(\alpha_{1},\alpha_{2},...,\alpha_{k})=\Delta (\alpha{1},\alpha{2},...,\alpha_{k})$称为Gram行列式.

​	定理$B-1$:欧式空间中向量$\alpha_{1},\alpha_{2},...,\alpha_{k}$的Gram矩阵$\Delta (\alpha_{1},\alpha_{2},...,\alpha_{k})$必为半正定矩阵,$\Delta (\alpha_{1},\alpha_{2},...,\alpha_{k})$为正定矩阵$\Leftrightarrow \alpha_{1},\alpha_{2},...,\alpha_{k}$线性无关.

​	证明:对向量组$\alpha_{1},\alpha_{2},...,\alpha_{k}$应用Schmidt正交化过程,得到一组正交向量组$\beta_{1},\beta_{2},...,\beta_{k}$,把$\alpha_{1},\alpha_{2},...,\alpha_{k}$用$\beta_{1},\beta_{2},...,\beta_{k}$表示如下:

<center>$$\alpha_{1}=\beta_{1}$$	$(B-1)$</center>

<center>$$\alpha_{2}=b_{12}\beta_{1}+\beta_{2}$$	$(B-1)$</center>

<center>$$...$$	$(B-1)$</center>

<center>$$\alpha_{k}=b_{1k}\beta_{1}+b_{2k}\beta_{2}+...+b_{k-1k}\beta_{k-1}+\beta_{k}$$	$(B-1)$</center>

<center>$$(\alpha_{1},\alpha_{2},...,\alpha_{k})=(\beta_{1},\beta_{2},...,\beta_{k})\begin{vmatrix} 1 & b_{12}  & ... & b_{1k}\\ 0 & 1 & ... & b_{2k}\\...  & ...  & ... & ... \\ 0&0  & ... & 1 \end{vmatrix}$$</center>

$$G=(\alpha_{1},\alpha_{2},...,\alpha_{k})=\begin{vmatrix} (\alpha_{1},\alpha_{1}) &(\alpha_{1},\alpha_{2})  & ... & (\alpha_{1},\alpha_{k})\\ (\alpha_{2},\alpha_{1}) &(\alpha_{2},\alpha_{2})  & ... & (\alpha_{2},\alpha_{k}) \\...  & ...  & ... & ... \\ (\alpha_{k},\alpha_{1}) &(\alpha_{k},\alpha_{2})  & ... & (\alpha{k},\alpha{k}) \end{vmatrix}=\left |\begin{pmatrix}\alpha_{1}^{T}\\\alpha_{2}^{T}\\...\\\alpha_{k}^{T}\end{pmatrix}\begin{pmatrix} \alpha_{1}&\alpha_{2}  &...  & \alpha_{k} \end{pmatrix}\right |$$

$$=\left |(\beta_{1},\beta_{2},...,\beta_{k})\begin{vmatrix} 1 & b_{12}  & ... & b_{1k}\\ 0 & 1 & ... & b_{2k}\\...  & ...  & ... & ... \\ 0&0  & ... & 1 \end{vmatrix}\right |^{T} *(\beta_{1},\beta_{2},...,\beta_{k})\begin{vmatrix} 1 & b_{12}  & ... & b_{1k}\\ 0 & 1 & ... & b_{2k}\\...  & ...  & ... & ... \\ 0&0  & ... & 1 \end{vmatrix}$$

$$=\left |\begin{vmatrix} 1 & b_{12}  & ... & b_{1k}\\ 0 & 1 & ... & b_{2k}\\...  & ...  & ... & ... \\ 0&0  & ... & 1 \end{vmatrix}^{T}\right |*\left|\begin{pmatrix}\beta_{1}^{T}\\\beta_{2}^{T}\\...\\\beta_{k}^{T}\end{pmatrix}\begin{pmatrix} \beta_{1}&\beta_{2}  &...  & \beta_{k} \end{pmatrix}\right|*\left|\begin{vmatrix} 1 & b_{12}  & ... & b_{1k}\\ 0 & 1 & ... & b_{2k}\\...  & ...  & ... & ... \\ 0&0  & ... & 1 \end{vmatrix}\right|$$

$$=\left|\begin{pmatrix}\beta_{1}^{T}\\\beta_{2}^{T}\\...\\\beta_{k}^{T}\end{pmatrix}\begin{pmatrix} \beta_{1}&\beta_{2}  &...  & \beta_{k} \end{pmatrix}\right|=G(\beta_{1},\beta_{2},...,\beta_{k})​$$

​	由$\beta_{1},\beta_{2},...,\beta_{k}​$的正交性可得$G(\alpha_{1},\alpha_{2},...,\alpha_{k})=G​$

<center>$$(\beta_{1},\beta_{2},...,\beta_{k})=\begin{vmatrix}(\beta_{1},\beta_{1})&  &  & \\ & (\beta_{2},\beta_{2}) &  & \\ &  & ... & \\ &  &  & (\beta_{k},\beta_{k})\end{vmatrix}=\prod_{i=1}^{k}\left |\beta_{i}  \right |^{2}\geq 0$$</center>

​	因为$\Delta (\alpha_{1},\alpha_{2},...,\alpha_{k})$的$k$阶主子阵也是Gram矩阵,所以行列式的值大于等于零,因此$\Delta$也是半正定矩阵.

​	$\forall k,G(\alpha_{1},\alpha_{2},...,\alpha_{k})\geq 0$,对于线性相关向量组$\alpha_{1},\alpha_{2},...,\alpha_{k}$,必有一向量是其余向量的线性组合,反之亦真.设$\alpha_{k}$是$\alpha_{1},\alpha_{2},...,\alpha_{k-1}$的线性组合,有:

<center>$$\alpha_{k}=\sum_{i=1}^{k-1}\alpha_{i}\alpha_{j}$$</center>

​	把$G(\alpha_{1},\alpha_{2},...,\alpha_{k})$的第$k$列减去第$i$列的$\alpha_{i}$倍$(i=1,2,...,k-1)$,由内积的线性性质有$(\alpha_{i},0)=0(i=1,2,...,k)$,故有:

<center>$$G(\alpha_{1},\alpha_{2},...,\alpha_{k})=\begin{vmatrix} (\alpha_{1},\alpha_{1}) &(\alpha_{1},\alpha_{2})  & ... & (\alpha_{1},0)\\ (\alpha_{2},\alpha_{1}) &(\alpha_{2},\alpha_{2})  & ... & (\alpha_{2},0) \\...  & ...  & ... & ... \\ (\alpha_{k},\alpha_{1}) &(\alpha_{k},\alpha_{2})  & ... & (\alpha_{k},0) \end{vmatrix}=0$$</center>

​	综上可得,$\Delta (\alpha_{1},\alpha_{2},...,\alpha_{k})$为正定矩阵$\Leftrightarrow \alpha_{1},\alpha_{2},...,\alpha_{k}$线性无关.

​	证毕.

​	

## 参考文献

【1】李航.《统计学习方法》.清华大学出版社.2012年3月



<center>$$Thanks$$</center>	

<center>$$Chilam$$</center>	

$$Ben$$