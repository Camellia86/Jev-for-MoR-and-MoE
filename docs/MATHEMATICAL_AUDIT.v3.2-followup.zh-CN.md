# v3.2 三步实验设计的数学审核

两个交叉评价、冻结 SCHED-A 拟合读出、E-G/E-C 匹配训练，分别检验部署路径上的参数差、固定生成模型上的读出增益、加入控制梯度后的训练效应。这三个对象具有明确的数学定义。原文中关于效应上界、交叉熵的方差偏好、样本量功效和严格同成本的部分结论超出了推导范围，需要按本审核修正。

审核对象为[原 v3.2 计划](EXPERIMENT_PLAN.v3.2.zh-CN.md)、[阶段一报告](PILOT_REPORT.v3.2-stage1.zh-CN.md)及[追加机制方案](EXPERIMENT_PLAN.v3.2-followup.zh-CN.md)。本文给出数学结论和执行前必须固定的定义。原阶段一未达到放行门槛的决定保留。

## 1. 三步设计的识别范围

### 1.1 交叉评价与训练原因

固定位置日程交换两套生成参数，可以识别相同执行路径分布上的参数差。需要同时固定日程端点、混合权重、文档、初始状态和缓存语义。第 4 节逐步推导四单元分解与交互项。

这项参数比较控制了部署路径。原 E 与 SCHED-A 的训练还同时改变了生成加权方式和控制损失，因此参数差本身不能进一步归因于控制损失。第三步通过匹配训练补足这一识别条件。

### 1.2 第三步的初始化定义

低秩适配（Low-Rank Adaptation，LoRA）把可训练增量写成两个低秩矩阵的乘积。下表中的 KL 指 Kullback–Leibler 散度（Kullback–Leibler Divergence），用于约束当前动作分布偏离参考分布的程度。

原 v3.2 阶段二与追加方案使用不同的初始化和读出，对应不同研究对象。

| 组成部分 | 原 v3.2 阶段二 | 追加机制方案 |
|---|---|---|
| 共同生成初始化 | LoRA 的 $B=0$ | 已训练的 $\theta_S$ |
| 冻结读出 | 零 LoRA 模型上重新拟合 | 冻结 $\theta_S$ 上拟合的读出 |
| KL 参考分布 | 零 LoRA 初始策略 | $\theta_S$ 初始策略 |
| 固定位置日程 | 零 LoRA 表上拟合 | $\theta_S$ 表上拟合 |
| 研究对象 | 从零适配时加入控制梯度 | 已适配模型继续训练时加入控制梯度 |

设训练算子为

$$
\theta_a^T=\mathcal A_a(\theta_0,H_0,p_{\rm ref},\xi),\qquad a\in\{G,C\},
$$

其中 $H_0$ 是冻结读出，$\xi$ 汇集数据顺序和随机抽样。只有两臂的 $\theta_0,H_0,p_{\rm ref},\xi$ 与优化配置相同，控制项开关才是被比较的训练干预。将这些共同输入由零 LoRA 改成 $\theta_S$，会改变训练效应本身。

这个差别直接进入 LoRA 梯度。对一层

$$
y=y_0+sBAx,\qquad u=\nabla_yF,
$$

微分为

$$
dF=u^\top dy
=s\,u^\top(dB)Ax+s\,u^\top B(dA)x.
$$

利用矩阵内积与迹的定义，分别识别两个系数：

$$
\nabla_BF=s\,u(Ax)^\top,\qquad
\nabla_AF=s\,B^\top u x^\top.
$$

当 $B=0$ 时，初始 $\nabla_AF=0$；在非零 $\theta_S$ 上，$A$ 已参与梯度更新。两种初始化的结果应分别命名。若第三步承接第二步已得到的冻结读出，追加方案检验的是表中右列的继续训练效应。

### 1.3 固定代理与完整训练效应

在快照 $\theta_t$ 上固定抽样历史 $h$ 与生成权重，定义

$$
G_{t,h}(\vartheta)=\frac1N\sum_{c,d}p_{cd}(\theta_t;h)L_{cd}(\vartheta).
$$

克隆相同优化器状态的一步比较，测量

$$
G_{t,h}(\theta_t+\Delta_C)-G_{t,h}(\theta_t+\Delta_G).
$$

对完整历史分布下的目标，令

$$
F(\theta)=\sum_h q_\theta(h)f_\theta(h).
$$

在允许交换求导与求和的条件下，由乘积法则逐项求导：

$$
\begin{aligned}
\nabla F(\theta)
&=\sum_h\big[q_\theta(h)\nabla f_\theta(h)+f_\theta(h)\nabla q_\theta(h)\big]\\
&=\mathbb E_{h\sim q_\theta}\!\left[
\nabla f_\theta(h)+f_\theta(h)\nabla\log q_\theta(h)\right].
\end{aligned}
$$

固定代理只保留其声明的导数路径。控制项补充当前行成本对读出的导数，历史分布和未来回报的全部导数仍需要另外处理。完整 E-G/E-C 训练后的差值包含参数、动作概率、抽样历史和优化器状态的反馈；它与固定代理的一步差值之间没有直接的数值恒等式。

遗漏未来成本确实能改变梯度。取两步过程，第一步动作 $d_1\sim\operatorname{Bernoulli}(q_\theta)$，$q_\theta=(1+e^{-\theta})^{-1}$。当前成本为 $\ell_1=0$，第二步成本为 $\ell_2=1-d_1$，且与第二步动作无关。总期望及其导数为

$$
F(\theta)=q_\theta\cdot0+(1-q_\theta)\cdot1=1-q_\theta,
\qquad
F'(\theta)=-\frac{e^{-\theta}}{(1+e^{-\theta})^2}
=-q_\theta(1-q_\theta).
$$

只用当前成本时，第一步的路由导数为 $0\cdot\nabla\log\Pr_\theta(d_1)=0$；第二步固定历史后，成本对当前动作是常数，其期望路由导数为该常数乘 $\sum_{d_2}\nabla p_\theta(d_2\mid d_1)=0$。因此局部路由导数为零，而有限 $\theta$ 下总目标导数严格为负。

诊断批次若同时用于求梯度和测更新后的损失，测量的是

$$
G_D(\theta+\Delta(D))-G_D(\theta).
$$

即使 $D$ 未参加此前的长程训练，这个量也应称为“固定诊断批次的单步代理下降”。独立泛化评价需要在 $D_{\rm update}$ 上构造更新，在另外的 $D_{\rm eval}$ 上测量。

## 2. 交叉熵读出的理论条件

### 2.1 加权条件均值

设 $X$ 为输入，$S$ 为随机软标签，$w>0$ 为样本权重，且 $0<\mathbb E_P[w]<\infty$。本实验的有限成本 softmax 标签各分量均为正。无参数约束、无正则的加权交叉熵（Cross-Entropy，CE）风险为

$$
\mathcal R(p)=\frac{\mathbb E_P[-w\sum_k S_k\log p_k(X)]}{\mathbb E_P[w]}.
$$

固定 $X=x$，分母与 $p(x)$ 无关。记 $a_k(x)=\mathbb E_P[wS_k\mid X=x]$，需最小化

$$
-\sum_k a_k(x)\log p_k(x),\qquad \sum_kp_k(x)=1.
$$

拉格朗日函数及一阶条件为

$$
\mathcal L=-\sum_k a_k\log p_k+\xi\left(\sum_kp_k-1\right),
\qquad -\frac{a_k}{p_k}+\xi=0.
$$

因此 $p_k=a_k/\xi$。对 $k$ 求和，并利用 $\sum_k S_k=1$，得到

$$
\xi=\sum_k a_k=\mathbb E_P[w\mid X=x],
\qquad
\boxed{p_k^\star(x)=\frac{\mathbb E_P[wS_k\mid X=x]}{\mathbb E_P[w\mid X=x]}}.
$$

本实验权重含当前行与全文的计分 token 数以及上一行深度枚举数。要约去权重，必须证明它在给定 $X$ 后可约去。用加权分布 $dQ=w\,dP/\mathbb E_P[w]$ 表示时，等价写法是 $p^\star(x)=\mathbb E_Q[S\mid X=x]$。

### 2.2 线性函数类与正则化

实际读出采用 $p=\operatorname{softmax}(WX+\beta)$，并惩罚 $\gamma\|W\|_F^2$。其驻点满足

$$
\mathbb E_Q[(p-S)X^\top]+2\gamma W=0,
\qquad \mathbb E_Q[p-S]=0.
$$

这些方程约束的是参数化函数类中的解。若在所有训练行都精确满足 $p_i=s_i$，交叉熵部分的参数梯度为零，于是 $2\gamma W=0$。在 $\gamma>0$ 时必须有 $W=0$，输出退化为由 $\beta$ 决定的常数分布。非恒定标签一般不能同时满足精确插值与这一正则化驻点条件。

读出目标的凸性则成立。对参数方向 $(U,v)$，令 $u_j=Ux_j+v$、$\alpha_j=w_j/\sum_kw_k$。二阶方向导数为

$$
\begin{aligned}
D^2F[(U,v),(U,v)]
&=\sum_j\alpha_j\left[\sum_d p_{jd}u_{jd}^2-
\left(\sum_d p_{jd}u_{jd}\right)^2\right]+2\gamma\|U\|_F^2\\
&=\sum_j\alpha_j\operatorname{Var}_{d\sim p_j}(u_{jd})+2\gamma\|U\|_F^2\ge0.
\end{aligned}
$$

$W$ 的严格凸惩罚保证其最优值唯一；$\beta$ 的公共平移保持概率不变。凸性支持最优性分析，具体数值算法的收敛仍由线搜索、终止条件和有限精度共同决定。原计划“凸优化不依赖步长”应改为“采用线搜索并核验最终梯度”。

### 2.3 部署等价中的拟合误差

本小节用 $\overline C_d=C_{cd}/n_c$ 表示每 token 成本，$C_{cd}$ 是第 5 节的行总成本，$n_c$ 是该行 token 数。对某一固定状态，若 $s_d=\operatorname{softmax}(-\overline C/\tau)_d$ 且 $p=s$，原计划的部署等价成立。一般情形定义

$$
e_d=\log p_d-\log s_d.
$$

代入部署得分并去掉与动作无关的归一化常数，得到

$$
\begin{aligned}
\arg\max_d[\log p_d-br_d]
&=\arg\max_d[-\overline C_d/\tau-\log Z+e_d-br_d]\\
&=\arg\min_d[\overline C_d+b\tau r_d-\tau e_d].
\end{aligned}
$$

所以 $b\tau$ 的成本单价解释成立于声明的代理目标；实际决策还受到动作相关的拟合误差 $-\tau e_d$ 影响。它也没有自动等于实际 FLOPs 账本下的损失斜率。

令 $A_d=\overline C_d+b\tau r_d$，$d^\star=\arg\min_d A_d$，$\widehat d=\arg\min_d(A_d-\tau e_d)$。由最优性不等式，

$$
A_{\widehat d}-\tau e_{\widehat d}\le A_{d^\star}-\tau e_{d^\star},
$$

从而

$$
0\le A_{\widehat d}-A_{d^\star}
\le\tau(e_{\widehat d}-e_{d^\star})
\le2\tau\|e\|_\infty.
$$

这是给定固定状态和成本后的条件性误差界。将缓存表示上的拟合误差界移到在线历史，还需要控制两种表示下的得分差。

### 2.4 随机成本与方差方向

二动作时，软标签满足

$$
S_2=\sigma\!\left(\frac{C_1-C_2}{\tau}\right),\qquad
\sigma''(u)=\sigma(u)(1-\sigma(u))(1-2\sigma(u)).
$$

二阶导数随 $u$ 改变符号，因此增加成本方差对期望软标签没有统一影响方向。原报告的反例正确地证明了软标签排序与期望成本排序可以冲突；它没有证明普遍的高方差偏好。

一个反方向的反例取 $\tau=0.05$、$C_1=0.3$。当 $C_2\equiv0.24$ 时，$S_2=\sigma(1.2)\approx0.7685$。将动作 2 的成本改为

$$
C_2=\begin{cases}0,&\text{概率 }0.4,\\0.4,&\text{概率 }0.6,\end{cases}
$$

则

$$
\mathbb E[C_2]=0.24,\qquad
\operatorname{Var}(C_2)=0.6(0.4)^2-(0.24)^2=0.0384.
$$

成本均值保持不变，方差增加，但

$$
\mathbb E[S_2]=0.4\sigma(6)+0.6\sigma(-2)\approx0.4705<0.5.
$$

交叉熵的无约束最优分类反而避开了期望成本较低的高方差动作。正确结论是：先对随机成本作 softmax 再取条件均值，与先取条件成本均值再决策，一般不等价。

### 2.5 拟合误差与表示信息

令 $q(X)=\mathbb E_Q[S\mid X]$。加入并减去 $\log q_k(X)$，再取条件期望：

$$
\begin{aligned}
\mathbb E_Q[\mathrm{KL}(S\Vert p(X))]
&=\mathbb E_Q\sum_k S_k\log\frac{S_k}{q_k(X)}
+\mathbb E_Q\sum_k S_k\log\frac{q_k(X)}{p_k(X)}\\
&=\mathbb E_Q[\mathrm{KL}(S\Vert q(X))]
+\mathbb E_Q\sum_k\mathbb E_Q[S_k\mid X]\log\frac{q_k(X)}{p_k(X)}\\
&=\mathbb E_Q[\mathrm{KL}(S\Vert q(X))]
+\mathbb E_Q[\mathrm{KL}(q(X)\Vert p(X))].
\end{aligned}
$$

第一项反映给定表示后的标签变化；第二项反映读出没有拟合到条件均值。观测总 KL 和交叉验证选择较强正则，无法分离这两项。线性函数类限制、正则偏差与有限样本误差都能贡献第二项。原报告据此判断“表示中可预测信息很少”应改为“当前线性读出与留出软标签之间的平均 KL 仍较大；该量包含条件标签变化与读出拟合误差”。

## 3. 统计结论与预算条件

### 3.1 不放行与效应上界

设真实增益为 $g$，点估计为 $\widehat g$，单侧名义 90% 下界为 $L_{90}$。原放行事件是

$$
A=\{\widehat g\ge0.005\}\cap\{L_{90}>0\}.
$$

没有观测到 $A$，只表示预定的证据条件没有同时满足。它不蕴含 $g<0.005$。阶段一交叉熵读出的 95% 区间为 [−0.0039, 0.0116]，表示岭回归为 [−0.0009, 0.0127]；两者都包含大于 0.006 的效应。

同一逻辑也适用于阶段二：联合训练未通过放行判据，支持“未达到预定证据标准”，真实联合训练增益的大小仍由该阶段的估计与区间描述。

原计划第 6 节的“增益低于 0.005”和原报告第 7 节的“增益不超过约 0.006”应替换为：

> 本次交叉熵读出未达到预定放行标准。区间同时包含零与实用门槛，现有证据尚未确定真实增益是否超过 0.005。

对某个已训练策略的区间，也不能转化为所有读取该表示的策略的收益上界。报告中表示岭回归相对交叉熵读出的直接配对差为 +0.0020，区间 [−0.0042, 0.0080]，因此“回归读出强于交叉熵读出”应改为“回归读出的改善点估计更大”。

### 3.2 放行概率随样本量的变化

在规划用的正态近似下，假定

$$
\widehat g\sim\mathcal N(\delta,\sigma^2/n),\qquad
L_{90}\approx\widehat g-z_{0.9}\sigma/\sqrt n.
$$

两个效应条件等价于

$$
\widehat g\ge\max\{\delta_{\min},z_{0.9}\sigma/\sqrt n\},\qquad\delta_{\min}=0.005.
$$

标准化后，放行概率为

$$
\Pr(A)\approx\Phi\!\left(
\frac{\delta-\max\{\delta_{\min},z_{0.9}\sigma/\sqrt n\}}{\sigma/\sqrt n}\right).
$$

当 $\delta>\delta_{\min}$ 时，$n\to\infty$ 使这个概率趋于 1；当 $\delta=\delta_{\min}$ 时趋于 $1/2$；当 $\delta<\delta_{\min}$ 时趋于 0。原报告“通过率被压在七成以下、增加样本只改善下界”不成立。

若目标功效为 $q>1/2$，需同时满足

$$
\delta\sqrt n/\sigma-z_{0.9}\ge z_q,
\qquad (\delta-\delta_{\min})\sqrt n/\sigma\ge z_q.
$$

在 $\delta>\delta_{\min}$ 时解得

$$
n\ge\max\left\{
\left[\frac{(z_{0.9}+z_q)\sigma}{\delta}\right]^2,
\left[\frac{z_q\sigma}{\delta-\delta_{\min}}\right]^2\right\}.
$$

代入报告的规划假设 $\delta=0.0058$、$\sigma=0.067$，378–1,500 篇对应的通过概率约为 0.59–0.68，达到 80% 约需 4,970 篇。这个数值取决于所假定的效应和标准差；从多个诊断中选出的较高点估计带有选择影响，重新划分数据并重训也会改变待评价策略。

### 3.3 逆方差合并的前提

若两项独立、无偏估计针对同一固定效应，方差分别为 $v_1,v_2$，则最小方差线性无偏合并为

$$
\widehat g_{\rm pool}=\frac{\widehat g_1/v_1+\widehat g_2/v_2}{1/v_1+1/v_2},
\qquad \operatorname{Var}(\widehat g_{\rm pool})=\frac1{1/v_1+1/v_2}.
$$

若协方差为 $c$，使用权重 $a,1-a$ 时，应有

$$
\operatorname{Var}(a\widehat g_1+(1-a)\widehat g_2)
=a^2v_1+(1-a)^2v_2+2a(1-a)c.
$$

原开发集已参与失败诊断并影响新版方法设计。文档不重叠不足以使其成为独立确认集。合并结果可用于描述两个数据集的观测表现，新的筛选集仍承担预先规定的确认任务。

### 3.4 成本容差与严格同成本

令两个锁定策略的实际账本成本为 $c_G,c_C$，观测差为

$$
g_{\rm obs}=L_G(c_G)-L_C(c_C).
$$

$c_C/c_G\in[0.98,1.02]$ 定义成本容差内的比较。若 $c_G\approx0.5$，它容许约 0.01 的绝对成本差。用负对数似然（Negative Log-Likelihood，NLL）衡量损失，假设同一策略族实际 NLL—成本曲线的下降斜率为 1.5–1.9，则这一成本差对应 0.015–0.019 nat/token，超过 0.005 的效应门槛。这是以所给斜率为条件的敏感性分析；训练的 $\lambda$、位置日程的 $\nu$ 和部署的 $\lambda+b\tau$ 不能直接替代这个实际斜率。

例如 $c_C\ge c_G$，记 $s_C(c)=-L_C'(c)$。积分得到

$$
\begin{aligned}
g_{\rm eq}(c_G)
&=L_G(c_G)-L_C(c_G)\\
&=g_{\rm obs}-\int_{c_G}^{c_C}s_C(u)\,du.
\end{aligned}
$$

若已知 $s_C\in[s_{\min},s_{\max}]$，则

$$
g_{\rm obs}-s_{\max}(c_C-c_G)
\le g_{\rm eq}(c_G)
\le g_{\rm obs}-s_{\min}(c_C-c_G).
$$

严格的共同成本比较可在预先指定的两端混合族内定义。对臂 $a$，两端成本为 $c_{a,-}<c_{a,+}$，相应损失为 $L_{a,-},L_{a,+}$，则

$$
L_a(c)=L_{a,-}+\frac{c-c_{a,-}}{c_{a,+}-c_{a,-}}(L_{a,+}-L_{a,-}).
$$

在两条曲线共同覆盖的 $B$ 上，比较 $g_B=L_G(B)-L_C(B)$。这是该混合族内的精确共同成本量。若在评价样本上估计成本与混合权重，bootstrap 的每次重采样都应重新计算成本、权重和 NLL。第 7 节的固定权重 bootstrap 对应原锁定策略的差值，两者应分别报告。

每次重采样还必须满足两臂各自的端点仍夹住 $B$。若某次算出 $w\notin[0,1]$，该公式成为线性外推，随机混合策略的解释随之失效。执行定义应预先规定端点夹逼失败的处理，例如报告失败比例并将该目标预算的区间标为不可估计。静默裁剪权重改变了目标预算；静默丢弃失败样本改变了重采样分布。两种操作都不能保留原统计量的解释。

执行者应在查看新筛选 NLL 前确定主统计量：保留原定义时，称为“成本容差内的锁定策略差值”；主张严格同成本优势时，另行预先固定共同成本估计方法。原阶段一判据继续按其原定义解释。

### 3.5 校准切换的条件

对固定得分 $a_d$，令 $d_1,d_2$ 分别是 $b_1<b_2$ 时 $a_d-br_d$ 的最大化动作。最优性给出

$$
a_{d_1}-b_1r_{d_1}\ge a_{d_2}-b_1r_{d_2},\qquad
a_{d_2}-b_2r_{d_2}\ge a_{d_1}-b_2r_{d_1}.
$$

相加后得到

$$
(b_2-b_1)(r_{d_1}-r_{d_2})\ge0,
$$

因此固定状态下所选代理成本随 $b$ 非增。在线多步决策中，得分还通过历史依赖 $b$，上述固定得分的证明不能直接推出整篇成本单调。

两动作的切换点满足

$$
b=\frac{\log p_j-\log p_k}{r_j-r_k},\qquad r_j\ne r_k.
$$

多个样本仍能有相同或接近的切换点。去除概率平滑消除了上下界强制造成的共同门槛；“不会出现成批切换”并无保证。只要实际找到跨过预算的两个在线端点，第 7 节的混合公式就成立。搜索能否找到这些端点需要按实际成本核验。

## 4. 交叉评价的可识别量

用 $\theta_S$ 表示原 SCHED-A 的 LoRA 参数，用 $\theta_E$ 表示 v3.1 E 的 LoRA。两套参数各自拟合的位置日程分别记为 $\pi_S$、$\pi_E$。先对确定性日程 $\rho$ 和文档 $i$ 定义在线执行的每 token NLL：

$$
L_i(\theta,\rho)=\frac{1}{N_i}\sum_{t\in\mathcal T_i}
-\log P_\theta\!\left(x_{i,t+1}\mid x_{i,\le t};\rho\right).
$$

$\mathcal T_i$ 是实现中的计分输入位置集合，$N_i=|\mathcal T_i|$。每个输入位置预测下一个 token。统计量先在文档内除以 token 数，再对文档取等权平均。

预算对齐后的日程 $\pi$ 在文档级以概率 $w$ 选择端点 $\rho_+$，以概率 $1-w$ 选择端点 $\rho_-$。评价量取该随机执行的期望损失：

$$
L_i(\theta,\pi)=wL_i(\theta,\rho_+)+(1-w)L_i(\theta,\rho_-),
\qquad
\overline L(\theta,\pi)=\frac1M\sum_{i=1}^M L_i(\theta,\pi).
$$

这个定义先对端点的预测概率取负对数，再对执行策略取期望。

原来的参数差写成

$$
D=\overline L(\theta_E,\pi_E)-\overline L(\theta_S,\pi_S).
$$

这个差同时改变两个变量。补齐交叉单元后，可以沿两条路径精确分解：

$$
\begin{aligned}
D
&=\big[\overline L(\theta_E,\pi_S)-\overline L(\theta_S,\pi_S)\big]
 +\big[\overline L(\theta_E,\pi_E)-\overline L(\theta_E,\pi_S)\big]\\
&=\big[\overline L(\theta_E,\pi_E)-\overline L(\theta_S,\pi_E)\big]
 +\big[\overline L(\theta_S,\pi_E)-\overline L(\theta_S,\pi_S)\big].
\end{aligned}
$$

第一行先换参数、再换日程。第二行先换日程、再换参数。两个固定日程下的参数效应为

$$
A_S=\overline L(\theta_E,\pi_S)-\overline L(\theta_S,\pi_S),
\qquad
A_E=\overline L(\theta_E,\pi_E)-\overline L(\theta_S,\pi_E).
$$

参数与日程的交互项定义为

$$
\begin{aligned}
I&=A_E-A_S\\
 &=\overline L(\theta_E,\pi_E)-\overline L(\theta_E,\pi_S)
   -\overline L(\theta_S,\pi_E)+\overline L(\theta_S,\pi_S).
\end{aligned}
$$

这四个单元确定了当前两套参数和两套日程之间的有限差分。本协议以教师强制方式输入相同的参考解答，位置日程只依赖行号和上一行长度，因此固定同一端点日程时，每篇文档的深度序列与计算成本在两套参数下相同。两端采用同一混合权重后，$A_S$、$A_E$ 比较相同执行路径分布上的生成质量。它们的差 $I$ 衡量参数效应随日程的变化。

这项交叉评价使用已经用于先导诊断的 505 篇文档。文档重采样给出以当前参数和日程为条件的开发性区间，文档参与过诊断选择这一事实仍需保留在解释中。E-G/E-C 的新检验使用另行保留的 378 篇文档。

## 5. 冻结读出的匹配训练目标

### 5.1 标准化表示与动作分布

动作集合为五个循环深度：

$$
\mathcal D=(4,6,8,12,16),\qquad
r=(0.301,0.418,0.534,0.767,1).
$$

训练缓存给出整段采用固定深度时的冻结循环核心特征。经过当前 LoRA 后，取第 $c$ 行前一位置的表示，记为 $\widetilde z_c(\theta;d_{c-1})$。冻结的均值与标准差分别为 $m$、$\sigma$，位置特征为 $\phi_c$。本节训练目标中的输入、logit 与动作概率定义为

$$
x_c(\theta)=\begin{bmatrix}(\widetilde z_c(\theta;d_{c-1})-m)\oslash\sigma\\\phi_c\end{bmatrix},
\qquad a_c=Wx_c+\beta,
\qquad p_{cd}=\frac{e^{a_{cd}}}{\sum_{j\in\mathcal D}e^{a_{cj}}}.
$$

$\oslash$ 表示逐元素相除。$W,\beta,m,\sigma$ 全部固定。LoRA 改变 $\widetilde z_c$，所以梯度仍能经过冻结读出回到 LoRA。

部署时，实际历史 $h_c$ 决定上一段末端的在线表示 $z_c^{\rm online}(\theta;h_c)$。把它代入同一标准化读出，得到 $p_c^{\rm online}$。部署规则为

$$
d_c(b)=\underset{d\in\mathcal D}{\arg\max}\,[\log p_{cd}^{\rm online}-b\,r_d].
$$

训练使用原始 softmax 概率 $p$。读出拟合时的正则项 $\gamma\|W\|_F^2$ 在这次匹配训练中是常数，梯度为零。

### 5.2 生成项与控制项

设当前文档第 $c$ 行有 $n_c$ 个计分 token，总数为 $N=\sum_c n_c$。$L_{cd}(\theta)$ 是固定核心特征经过当前 LoRA 后，该行在深度 $d$ 的 NLL 总和。记停止梯度算子为 $\operatorname{sg}$。生成目标为

$$
G(\theta)=\frac1N\sum_c\sum_d
\operatorname{sg}(p_{cd})L_{cd}(\theta).
$$

给定已抽样的历史 $h_c$，当前行成本为

$$
C_{cd}=\operatorname{sg}(L_{cd})
+n_c\big[\lambda r_d+\kappa_s q_s(h_c,d)+\kappa_e q_e(h_c,d)\big].
$$

$q_s$、$q_e$ 分别是键值缓存（Key-Value Cache，KV cache）的覆盖不足量与覆盖超出量。记此前解答行 $j$ 的实际计算位置数为 $m_j$，执行深度为 $d_j$，则

$$
q_s(h_c,d)=\frac{\sum_{j<c}m_j(d-d_j)_+}{\sum_{j<c}m_j},
\qquad
q_e(h_c,d)=\frac{\sum_{j<c}m_j(d_j-d)_+}{\sum_{j<c}m_j},
\qquad (u)_+=\max(u,0).
$$

这些量以此前解答位置作平均，第一行的空历史规定二者均为零，题目段不计入分母。每循环四个核心层，因此相应缺失的“位置—核心层”条目数为 $4\sum_{j<c}m_j(d-d_j)_+$。$m_j$ 是计算位置数，$n_j$ 是计分 token 数，两者分别用于缓存覆盖和损失归一化。

此次固定 $\lambda=1$，沿用原估计的完整精度 $\kappa_s=0.013001121873651982$、$\kappa_e=-0.0022825558298829535$。这是已拟合的局部成本修正，关于它的梯度推导以系数固定为条件；在线策略质量仍由在线 NLL 评价。软标签与参考分布为

$$
s_{cd}=\operatorname{sg}\!\left[
\frac{\exp(-C_{cd}/(n_c\tau))}{\sum_j\exp(-C_{cj}/(n_c\tau))}\right],
\qquad \tau=0.05,
\qquad p^{\rm ref}_{cd}=p_{cd}(\theta_S;d_{c-1}).
$$

参考分布在更新前计算并固定，按文档、行号和上一行深度索引。只对 $n_c>0$ 的行构造损失。控制目标为

$$
\begin{aligned}
Q(\theta)=\frac1N\sum_c\Bigg[
 &\sum_d p_{cd}C_{cd}
 -0.1n_c\sum_d s_{cd}\log p_{cd}\\
 &+0.01n_c\sum_d(p_{cd}-s_{cd})^2
 +0.1n_c\sum_d p_{cd}\log\frac{p_{cd}}{p^{\rm ref}_{cd}}
\Bigg].
\end{aligned}
$$

E-G 优化 $G$；E-C 优化 $G+Q$。每个批次包含八篇文档，批次目标是这八个文档目标的算术平均。两臂从同一非零 $\theta_S$、新的 AdamW 状态、相同文档排列与相同历史采样种子开始。

停止梯度规定了每次更新的局部求导对象。在当前参数 $\theta_t$ 上先计算 $p^t,C^t,s^t$，再把它们放入各自被停止梯度的位置，得到关于候选参数 $\vartheta$ 的普通函数 $G_t(\vartheta)$ 与 $Q_t(\vartheta)$。下节的梯度就是这两个函数在 $\vartheta=\theta_t$ 处的导数。每步重新生成标签与权重后，局部函数也随之变化，因此以下等式应按这一求导约定解释。

第零步两臂的参数、读出、历史和生成权重相同。后续参数改变后，动作概率以及抽样历史随之改变。因此，两臂最终在线差值估计的是加入控制梯度的训练总效应，包含此后生成权重和历史分布的变化。

### 5.3 控制梯度的逐项推导

固定一行，省略行下标。softmax 的雅可比矩阵为

$$
\frac{\partial p_i}{\partial a_k}=p_i(\mathbf1_{i=k}-p_k).
$$

期望成本项的导数由求和展开得到：

$$
\begin{aligned}
\frac{\partial}{\partial a_k}\sum_i p_i C_i
&=\sum_i C_i p_i(\mathbf1_{i=k}-p_k)\\
&=p_kC_k-p_k\sum_i p_iC_i\\
&=p_k(C_k-\mathbb E_p[C]).
\end{aligned}
$$

交叉熵使用固定软标签。由于 $\sum_i s_i=1$，

$$
\begin{aligned}
\frac{\partial}{\partial a_k}\left[-\sum_i s_i\log p_i\right]
&=-\sum_i s_i(\mathbf1_{i=k}-p_k)\\
&=-s_k+p_k\sum_i s_i\\
&=p_k-s_k.
\end{aligned}
$$

Brier 项的导数为

$$
\begin{aligned}
\frac{\partial}{\partial a_k}\sum_i(p_i-s_i)^2
&=2\sum_i(p_i-s_i)p_i(\mathbf1_{i=k}-p_k)\\
&=2p_k\left[(p_k-s_k)-\sum_i p_i(p_i-s_i)\right].
\end{aligned}
$$

对 KL 项，先对概率求导，再乘 softmax 雅可比：

$$
\begin{aligned}
\frac{\partial}{\partial a_k}\sum_i p_i\log\frac{p_i}{p_i^{\rm ref}}
&=\sum_i\left(\log\frac{p_i}{p_i^{\rm ref}}+1\right)
 p_i(\mathbf1_{i=k}-p_k)\\
&=p_k\left[\log\frac{p_k}{p_k^{\rm ref}}+1
 -\sum_i p_i\log\frac{p_i}{p_i^{\rm ref}}-\sum_i p_i\right]\\
&=p_k\left[\log\frac{p_k}{p_k^{\rm ref}}
 -\mathrm{KL}(p\Vert p^{\rm ref})\right].
\end{aligned}
$$

将四项按系数相加，定义第 $c$ 行的 logit 梯度 $v_c$：

$$
\begin{aligned}
v_{ck}=\frac1N\Bigg\{
 &p_{ck}(C_{ck}-\mathbb E_{p_c}[C_c])+0.1n_c(p_{ck}-s_{ck})\\
 &+0.02n_cp_{ck}\left[(p_{ck}-s_{ck})-\sum_i p_{ci}(p_{ci}-s_{ci})\right]\\
 &+0.1n_cp_{ck}\left[\log\frac{p_{ck}}{p^{\rm ref}_{ck}}
 -\mathrm{KL}(p_c\Vert p_c^{\rm ref})\right]\Bigg\}.
\end{aligned}
$$

写 $J_c=\partial a_c/\partial\theta$，两个参数梯度为

$$
g_G=\frac1N\sum_{c,d}p_{cd}\nabla_\theta L_{cd},
\qquad g_Q=\sum_c J_c^\top v_c.
$$

这些等式对应代码里的停止梯度位置。历史采样分布对参数的导数没有进入该局部目标。若改为对完整历史分布下的期望总损失求导，还需要历史概率的导数项；本实验的梯度诊断只针对上面明确给出的代理目标。

### 5.4 生成梯度与局部期望成本梯度

固定历史后，记与参数无关的成本惩罚为

$$
K_{cd}=n_c[\lambda r_d+\kappa_s q_s(h_c,d)+\kappa_e q_e(h_c,d)].
$$

去掉辅助项，定义普通可微的当前行期望成本

$$
F_h(\theta)=\frac1N\sum_{c,d}p_{cd}(\theta)[L_{cd}(\theta)+K_{cd}].
$$

乘积法则给出

$$
\begin{aligned}
\nabla F_h(\theta)
&=\frac1N\sum_{c,d}\left\{
p_{cd}\nabla L_{cd}+(L_{cd}+K_{cd})\nabla p_{cd}\right\}\\
&=g_G+g_{\rm route},
\end{aligned}
$$

其中 $g_{\rm route}$ 是第 5.3 节期望成本项传回参数的梯度。因此，生成项对 $p$ 停止梯度、路由项对 $L$ 停止梯度，这两条导数路径恰好拼成固定历史下 $F_h$ 的完整梯度。两个代理项的数值之和含有两份 $\sum pL$，却只各自保留乘积法则的一条导数路径；分析时应依据导数路径解释它们。

辅助项的软标签也采用停止梯度，其导数按第 5.2 节的局部函数定义处理。若把固定历史推广为参数相关的随机历史，还须加入第 1.3 节推导的历史概率导数。

## 6. 梯度与 AdamW 单步的解释

### 6.1 相同优化器状态上的比较

欧氏内积 $g_G^\top g_Q$ 衡量两个当前梯度的方向关系。在无动量、无预条件、无权重衰减的梯度下降中，联合更新为 $\Delta=-\eta(g_G+g_Q)$，于是

$$
g_G^\top\Delta=-\eta\|g_G\|_2^2-\eta g_G^\top g_Q.
$$

这个等式说明负内积会削弱该理想更新的生成下降量。本实验使用 AdamW，实际更新还依赖累积的一阶矩与二阶矩。

记裁剪后梯度为 $\widetilde g$、步长为 $\eta_t$、权重衰减为 $\omega$。同一快照上的两条更新分支分别使用 $g_G$ 与 $g_G+g_Q$。对每条分支，

$$
\begin{aligned}
\widetilde g&=g\min\!\left(1,\frac1{\|g\|_2+10^{-6}}\right),\\
m_t&=\beta_1m_{t-1}+(1-\beta_1)\widetilde g,\\
u_t&=\beta_2u_{t-1}+(1-\beta_2)\widetilde g^{\odot2},\\
\widehat m_t&=m_t/(1-\beta_1^t),\qquad
\widehat u_t=u_t/(1-\beta_2^t),\\
\Delta_t&=-\eta_t\omega\theta
-\eta_t\widehat m_t\oslash(\sqrt{\widehat u_t}+\epsilon).
\end{aligned}
$$

默认系数为 $\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$，此次 $\omega=0.01$。两分支克隆相同的 $m_{t-1}$、$u_{t-1}$、步数及学习率。分别执行真实优化器更新后，记录 $g_G^\top\Delta_G$ 与 $g_G^\top\Delta_{G+Q}$。

生成目标中的停止梯度权重需要在单步比较时固定。令当前快照参数为 $\theta_0$，保存该次前向使用的动作历史和概率 $p^0$，定义

$$
G_0(\theta)=\frac1N\sum_{c,d}p^0_{cd}L_{cd}(\theta).
$$

实际单步变化为

$$
\delta_G=G_0(\theta_0+\Delta_G)-G_0(\theta_0),
\qquad
\delta_{G+Q}=G_0(\theta_0+\Delta_{G+Q})-G_0(\theta_0).
$$

令 $G_0^{\rm real}$ 为采用理想实数运算的对应目标。若它在更新线段邻域二次连续可微，记 $g_G^{\rm real}=\nabla G_0^{\rm real}(\theta_0)$、$H_0^{\rm real}=\nabla^2G_0^{\rm real}$，泰勒展开给出

$$
G_0^{\rm real}(\theta_0+\Delta)-G_0^{\rm real}(\theta_0)
=(g_G^{\rm real})^\top\Delta+\frac12\Delta^\top H_0^{\rm real}(\theta_0+\xi\Delta)\Delta,
\qquad 0<\xi<1.
$$

实际前向采用 16 位脑浮点格式（Brain Floating Point 16，BF16），参数投影与中间结果带有量化舍入。代码中的 $g_G$ 是该计算图按自动微分规则得到的梯度。实测损失变化与 $g_G^\top\Delta$ 的差同时受到有限步长和数值舍入影响。诊断分别报告一阶内积和重新前向得到的损失差。固定代理的一步变化衡量局部训练机制，最终在线 NLL 衡量经过完整训练后的策略表现。

### 6.2 首次 AdamW 更新中的正交反例

首次更新的矩状态为零。对任意待更新梯度 $u$，设裁剪系数为 $\alpha_u>0$。偏差修正后，两个矩为

$$
\widehat m_1=\alpha_u u,\qquad
\widehat u_1=(\alpha_u u)^{\odot2}.
$$

代入 AdamW 更新式，得到理想实数运算下的精确表达式：

$$
\Delta(u)=-\eta\omega\theta
-\eta\,\frac{\alpha_u u}{\alpha_u|u|+\epsilon}.
$$

这里分式逐坐标计算。令 $u_G=g_G$、$u_J=g_G+g_Q$。两分支的初始参数、步长和衰减系数相同，衰减项在位移之差中抵消。因此，

$$
g_G^\top(\Delta_J-\Delta_G)
=\eta\sum_i g_{G,i}\left[
\frac{\alpha_Gg_{G,i}}{\alpha_G|g_{G,i}|+\epsilon}
-\frac{\alpha_Ju_{J,i}}{\alpha_J|u_{J,i}|+\epsilon}\right].
$$

进一步取 $\epsilon\to0$，并假设所比较坐标的 $g_{G,i}$、$u_{J,i}$ 均非零。逐项取极限后，

$$
\begin{aligned}
\lim_{\epsilon\to0}g_G^\top(\Delta_J-\Delta_G)
&=\eta\sum_i\left[|g_{G,i}|-g_{G,i}\operatorname{sign}(u_{J,i})\right]\\
&=2\eta\sum_{i:\,\operatorname{sign}(g_{G,i})\ne\operatorname{sign}(u_{J,i})}|g_{G,i}|\\
&\ge0.
\end{aligned}
$$

纯生成梯度在这个极限下给出梯度驱动位移 $-\eta\operatorname{sign}(g_G)$。对所有满足 $\|v\|_\infty\le\eta$ 的位移，逐坐标有 $g_{G,i}v_i\ge-\eta|g_{G,i}|$。求和得到

$$
g_G^\top v\ge-\eta\|g_G\|_1,
$$

且纯生成的符号更新达到等号。这解释了上面的非负差值：在这一坐标盒约束内，纯生成梯度已达到最大的线性下降。

一个严格正交的例子是

$$
g_G=(1,1),\qquad g_Q=(-2,2),\qquad g_G^\top g_Q=0.
$$

联合梯度为 $(-1,3)$。在符号更新极限下，去除公共衰减项后的位移分别为 $v_G=(-\eta,-\eta)$、$v_J=(\eta,-\eta)$，所以

$$
g_G^\top v_G=-2\eta,\qquad g_G^\top v_J=0.
$$

这个反例证明正交不足以保证生成下降量保持不变。实际 $\epsilon=10^{-8}$ 对应上面的有限 $\epsilon$ 表达式；接近符号极限还要求相关坐标的裁剪后幅度远大于 $\epsilon$。后续更新具有非零矩状态，适用第 6.1 节的完整公式。训练影响由实际更新与独立在线评价分别衡量。

## 7. 成本对齐与配对区间

计算成本采用计划的浮点运算次数（Floating-Point Operations，FLOPs）账本，按每篇解答行的全部输入位置累计，排除问题前缀。设逐位置计算账本为 $f(t,d)$，第 $c$ 行的位置区间为 $[a_{ic},b_{ic})$，确定性日程的相对成本为

$$
c(\rho)=\frac{\sum_i\sum_c\sum_{t=a_{ic}}^{b_{ic}-1}f(t,d_{ic}(\rho))}
{\sum_i\sum_c\sum_{t=a_{ic}}^{b_{ic}-1}f(t,16)}.
$$

损失只统计 $\mathcal T_i$ 中有下一 token 标签的位置；计算成本统计整行实际执行的位置。五维向量 $r$ 用于训练代理和决策惩罚，预算对齐使用这里的逐位置成本账本。

对校准集上的两个策略，设成本满足 $c_-\le B\le c_+$，且 $c_+>c_-$。在文档级以概率 $w$ 选择高成本策略，则期望成本为

$$
c(w)=wc_++(1-w)c_-=c_-+w(c_+-c_-).
$$

令 $c(w)=B$，解得

$$
w=\frac{B-c_-}{c_+-c_-}.
$$

两端策略均真实在线执行。每篇文档的混合 NLL 直接取两端结果的加权和，等于该随机策略的精确期望。筛选集沿用校准集固定的 $w$；筛选集成本重新测量。

E-G 与 E-C 各自校准后的主差定义为

$$
g_{\rm joint}=\overline L(\theta_G,\pi_G)-\overline L(\theta_C,\pi_C).
$$

在同一固定位置日程 $\pi_R$ 下，生成质量差定义为

$$
h_{\rm gen}=\overline L(\theta_C,\pi_R)-\overline L(\theta_G,\pi_R).
$$

$g_{\rm joint}>0$ 表示 E-C 的整体在线策略更好；$h_{\rm gen}>0$ 表示 E-C 在同一路径上的生成 NLL 更高。两个量的方向约定不同，分别对应整体改善与固定路径退化。

对任一逐篇配对差 $d_i$，每次有放回抽取 $M$ 个文档索引，计算

$$
\widehat d^{*(b)}=\frac1M\sum_{j=1}^M d_{I_j^{(b)}},
\qquad I_j^{(b)}\overset{\mathrm{iid}}\sim\operatorname{Unif}\{1,\ldots,M\}.
$$

固定种子 20261013，重复 10,000 次。双侧名义 95% 区间取重采样均值的 2.5% 与 97.5% 分位数；单侧名义 90% 下界取 10% 分位数。这些分位数用文档重采样估计当前训练种子及已锁定校准参数下的文档抽样不确定性。跨训练种子的变异需要独立训练重复来估计。

## 8. 执行交接的数学约束

三步顺序能够分别回答部署路径、冻结读出和联合训练的问题。执行结果应使用对应的效应定义，避免把其中一步的结论推广成另一阶段的结论。

1. **交叉评价固定整个日程。** 交换生成参数时，保留端点、混合权重、深度序列与计分位置。第 4 节的两个参数效应与交互项均需报告。交互项的区间包含零，只表示当前数据尚未区分它与零；主张交互足够小需要先给出等效界，再检验整个区间是否位于该界内。

2. **冻结读出明确数据与参数来源。** 冻结 $\theta_S$、拟合读出、选择正则与校准部署参数，各步骤沿用其预先划定的数据角色。第二步的目标是固定 $\theta_S$ 时读出相对位置日程的增益。已有读出的复用属于已有结果的核验，独立重复需要重新声明重复单位与抽样方式。

3. **匹配训练明确初始化。** 沿用追加方案时，两臂共同从 $\theta_S$ 开始，冻结同一个读出，使用相同参考分布、初始优化器状态、文档顺序与随机数规则。此时结论限于已适配模型的继续训练。原 v3.2 阶段二的零 LoRA 初始化定义另一个效应，其结果需要单独获得。

4. **单步诊断固定待测函数。** 每个快照先保存历史与生成权重，然后克隆相同优化器状态。两分支在完全相同的 $G_0$ 上测量一阶内积和实际变化。采用同一批次求更新与测量时，报告名称使用“固定诊断批次的单步代理下降”。

5. **成本与区间对应同一个统计量。** 原追加方案的主量是第 7 节的锁定策略差值，并附成本比容差条件。若执行者增加严格共同预算的比较，应在读取新筛选 NLL 前固定预算、端点、权重估计方法、重采样方法和夹逼失败处理，并单列为第 3.4 节的统计量。成本比通过容差与严格同成本优势应分别表述。

6. **共同随机种子限定效应解释。** 现有单种子设计识别固定训练随机性下的两臂差值，逐篇配对区间衡量文档抽样不确定性。对训练算法跨种子的平均收益下结论，需要增加独立训练重复并在相应层级计算不确定性。

原 v3.2 阶段二还要求一个追加方案尚未包含的量：E-C 相对其自身位置基线的增益。为区分两个位置日程，记在 $\theta_C$ 上按原配方拟合并校准的日程为 $\pi_{R,C}$，则

$$
g_C=\overline L(\theta_C,\pi_{R,C})-\overline L(\theta_C,\pi_C).
$$

这里保持生成参数为 $\theta_C$，只比较读出策略与该生成模型自己的位置基线。第 7 节的 $h_{\rm gen}$ 使用同一个预先固定的 $\pi_R$ 比较两套生成参数，变量改变方式不同。

$g_{\rm joint}$ 与 $g_C$ 满足如下恒等式：

$$
\begin{aligned}
g_{\rm joint}
&=\overline L(\theta_G,\pi_G)-\overline L(\theta_C,\pi_C)\\
&=\big[\overline L(\theta_G,\pi_G)
       -\overline L(\theta_C,\pi_{R,C})\big]+g_C.
\end{aligned}
$$

方括号中的差改变了参数和策略，$h_{\rm gen}$ 也没有给出 $\overline L(\theta_C,\pi_{R,C})$。因此，已有 $g_{\rm joint}$ 和 $h_{\rm gen}$ 不能单独解出 $g_C$。若交付目标包含原阶段二的完整三个评价量，执行者需补充 $\pi_{R,C}$ 的独立拟合、预算校准与评价；若仅执行追加方案，应将结论限定在其已经定义的两个量。

数学审核结论是：三步设计可用于上述条件明确的机制比较；加权交叉熵最优性、控制梯度和 AdamW 的推导按本文条件成立。统计结论应保留区间与成本定义，阶段一未放行的原决定保持有效。
