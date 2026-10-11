# 共享深度控制的数学审计

本文对应[实验计划 v1](EXPERIMENT_PLAN.zh-CN.md)。推导对象是一次前缀决策、四个固定循环深度、共享生成与控制末端的模型。解析结论、数值代数检查和真实模型实验分开列出：文末的 CPU 检查已经运行，真实模型的因果、缓存、显存、质量和速度检查仍待先导完成。

## 1. 计算图与合法信息

### 1.1 模型定义

令完整教师强制输入为：

$$
u=(x_1,\ldots,x_L,y_1,\ldots,y_{H-1}),\qquad L=H=128.
$$

冻结的前置模块记为 $E$，循环核心记为 $F$，共享末端记为 $G_\theta$，冻结词表投影记为 $V$。核心初始状态 $\xi_{i,t}$ 由文档来源行号、绝对位置和独立初始化种子生成。给定这些随机量后，模型前向是确定函数：

$$
e_{1:n}=E(u_{1:n}),
$$

$$
r^{(0)}_{1:n}=\xi_{i,1:n},\qquad
r^{(k+1)}_{1:n}=F_k(r^{(k)}_{1:n},e_{1:n}).
$$

这里 $F_k$ 共用可学习权重，下标 k 区分展开后的层位置与缓存索引。令 N 为冻结归一化层，两个末端块合记为 G：

$$
a^{(d)}=N(r^{(d)}),\qquad
z^{(d)}_\theta=N(G_\theta(a^{(d)})).
$$

原模型同一个归一化模块 N 在核心出口和末端出口各调用一次。缓存 $a^{(d)}$ 后直接送入 G；若缓存的是 $r^{(d)}$，则先运行 N。两种缓存格式必须有不同类型标记。

目标位置的语言分布为：

$$
P_\theta(y_j\mid x,y_{<j};d)
=\operatorname{softmax}\!\left(Vz^{(d)}_{\theta,L+j-1}\right)_{y_j}.
$$

控制器取独立前缀运行的末端表示：

$$
p_\theta(d\mid x)
=\operatorname{softmax}\!\left(W_0z^{(4)}_{\theta,L}+b_0\right)_d.
$$

因此生成概率与路由概率都依赖同一个 $\theta$。在训练后冻结 $W_0,b_0$，仍然能通过共享末端改变路由。

### 1.2 因果性证明

假设每个注意力节点只读本位置及其之前的位置，每个 token 的初始状态不依赖后续文本，归一化和前馈模块逐位置计算。对展开层级和 token 位置作归纳。

第一，输入 embedding 和初始状态在位置 t 仅依赖 $u_t$、行号、位置与随机种子，故不依赖 $u_{>t}$。

第二，假设某一层之前所有位置 $s\le t$ 的状态仅依赖 $u_{\le s}$。该层位置 t 的注意力只聚合 $s\le t$ 的键和值，所以聚合结果只依赖 $u_{\le t}$。残差、逐位置归一化和前馈变换保持这个性质。

第三，对所有前置层、循环展开层和末端层重复第二步，得到：

$$
z^{(d)}_{\theta,t}
=\phi_{\theta,d,t}(u_{\le t},\xi_{i,\le t}).
$$

令 $t=L$，可得前缀控制概率与真实后缀无关：

$$
p_\theta(d\mid x,y)=p_\theta(d\mid x).
$$

训练用后缀形成成本 $C_\theta(d)$，然后用该成本监督前缀控制器，是合法的监督学习。推理控制器的输入仍然只有 x。

这个证明有一个容易被实现破坏的条件：若 $\xi_{i,t}$ 的 seed 是全文哈希，则 $\xi_{i,t}$ 会随未来后缀改变。注意力掩码仍然是因果的，但控制器的随机输入已经携带后缀函数。协议因此把全文哈希限定于数据去重与切分，模型初始化只使用读取文本前已确定的行号和位置。

### 1.3 暂停续算与增量生成

循环核心在第 k 轮的充分状态为原始 $r^{(k)}$、前置表示 e、绝对位置、展开层号、下一循环编号和对应 KV。保持这些变量，先运行 4 轮再运行 $d-4$ 轮与连续运行 d 轮具有完全相同的函数复合：

$$
F_{d-1}\circ\cdots\circ F_4
\left(F_3\circ\cdots\circ F_0(r^{(0)},e),e\right)
=F_{d-1}\circ\cdots\circ F_0(r^{(0)},e).
$$

等号左侧使用的是原始循环状态。插入一次额外归一化得到的是 $F_4(N(r^{(4)}),e)$，通常与 $F_4(r^{(4)},e)$ 不同；因此归一化后的 latent 不能冒充续算状态。

同理，固定深度 d、逐展开层独立的 full KV 策略下，整段教师强制与逐 token 前向是同一有向无环因果图的两种求值顺序。逐节点看，两者提供相同的本层输入和历史 K/V，因此在实数运算中输出相同。

BF16 的求和顺序受矩阵形状影响，数学上的函数等价不意味着浮点逐位相等。实验必须用相同显式初始状态核验 raw state、目标概率和 NLL 的差异。协议的阈值在正式实验前固定。

## 2. Dirichlet 期望目标的解析化

### 2.1 均值

有 K=4 个动作。令 $p=\operatorname{softmax}(z)$，参数为：

$$
\alpha_d=\kappa\left[(1-\varepsilon)p_d+\frac{\varepsilon}{K}\right]+\eta,
\qquad \kappa=256,\quad\varepsilon=0.01,\quad\eta=0.001.
$$

由于 $\sum_dp_d=1$，总浓度与 z 无关：

$$
A=\sum_d\alpha_d
=\kappa\left[(1-\varepsilon)\sum_dp_d+\varepsilon\right]+K\eta
=\kappa+K\eta.
$$

设 $q\sim\operatorname{Dirichlet}(\alpha)$。Dirichlet 的均值为：

$$
\mathbb E[q_d]=\frac{\alpha_d}{A}
=\frac{\kappa(1-\varepsilon)}{\kappa+K\eta}p_d
+\frac{\kappa\varepsilon/K+\eta}{\kappa+K\eta}.
$$

定义：

$$
\beta=\frac{\kappa(1-\varepsilon)}{\kappa+K\eta}.
$$

整理常数项可得：

$$
\mu_d:=\mathbb E[q_d]=\beta p_d+\frac{1-\beta}{K}.
$$

所有动作概率均为正；$\eta$ 和 $\varepsilon$ 的作用是概率平滑。浓度还影响 q 的方差，但不会产生新的动作结果信息。

### 2.2 固定成本下的期望与梯度

先把成本向量 C 看成与参数无关的常数，定义线性成本：

$$
v(q)=q^\top C,\qquad
J=\mathbb E_q[v(q)].
$$

利用期望的线性性：

$$
J=\sum_dC_d\mathbb E[q_d]
=\sum_d\mu_dC_d
=\beta\sum_dp_dC_d+\frac{1-\beta}{K}\sum_dC_d.
$$

softmax 的导数是：

$$
\frac{\partial p_d}{\partial z_j}=p_d(\mathbf1\{d=j\}-p_j).
$$

代入并展开：

$$
\begin{aligned}
\frac{\partial J}{\partial z_j}
&=\beta\sum_dC_dp_d(\mathbf1\{d=j\}-p_j)\\
&=\beta C_jp_j-\beta p_j\sum_dC_dp_d\\
&=\beta p_j(C_j-p^\top C).
\end{aligned}
$$

所有成本相同时，括号为零，路由梯度为零。对所有动作同时加同一个常数 a，也有：

$$
(C_j+a)-\sum_dp_d(C_d+a)=C_j-p^\top C,
$$

所以路由梯度对共同平移不变。这是梯度实现的必要检查。

### 2.3 先抽概率再抽动作

如果先采样 q，再从 q 中采一个动作 D，则：

$$
\Pr(D=d\mid x)
=\int q_d\,\operatorname{Dir}(q;\alpha)\,dq
=\mu_d.
$$

因此单次部署可以直接从 $\operatorname{Categorical}(\mu)$ 采样，得到完全相同的动作边缘分布。该等价不适用于“从 q 取 argmax”或一次 q 控制多个相关动作；这些规则定义了其他目标。

主实验已经算出全部四动作成本，所以 E 能直接求期望。若未来只能执行一个动作并观察其结果，普通 categorical 策略梯度也必须列为比较项；届时反馈条件已经改变。

## 3. 共享参数要求两条梯度

### 3.1 联合目标

共享末端使 $\mu_\theta$ 与 $C_\theta$ 同时依赖参数。定义：

$$
J(\theta)=\sum_d\mu_\theta(d)C_\theta(d),
\qquad C_\theta(d)=L_\theta(d)+\lambda f(d).
$$

f 是固定执行形状下的计算成本，与 LoRA 的数值无关。对乘积逐项求导：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_d\nabla_\theta\!\left[\mu_\theta(d)C_\theta(d)\right]\\
&=\sum_dC_\theta(d)\nabla_\theta\mu_\theta(d)
+\sum_d\mu_\theta(d)\nabla_\theta C_\theta(d)\\
&=\underbrace{\sum_dC_\theta(d)\nabla_\theta\mu_\theta(d)}_{\text{路由梯度}}
+\underbrace{\sum_d\mu_\theta(d)\nabla_\theta L_\theta(d)}_{\text{生成梯度}}.
\end{aligned}
$$

只保留第一项，训练的是当前成本表上的深度偏好；第二项才使被选中深度的生成能力改善。两项都应回传至共享末端。

### 3.2 可检验的标量反例

取两个动作，令：

$$
\mu_1(\theta)=\sigma(\theta),\quad\mu_2(\theta)=1-\sigma(\theta),
$$

$$
C_1(\theta)=(\theta-1)^2,\quad C_2(\theta)=2.
$$

在 $\theta=0$ 时：

$$
\mu_1=\mu_2=\frac12,\quad
\mu'_1=\frac14,\quad\mu'_2=-\frac14,\quad
C_1=1,\quad C'_1=-2.
$$

所以路由项和生成项分别是：

$$
g_{\rm route}=\frac14\cdot1-\frac14\cdot2=-\frac14,
$$

$$
g_{\rm direct}=\frac12\cdot(-2)+\frac12\cdot0=-1.
$$

完整梯度为：

$$
J'(0)=-\frac14-1=-1.25.
$$

固定旧成本表只得到 -0.25，遗漏了本例 80% 的梯度幅度。更一般地，旧成本还会改变动作排序：本例在 $\theta=3$ 时 $C_1=4>C_2=2$，而初始表始终声称 $C_1=1<C_2=2$。因此，冻结核心特征与冻结动作损失是两个不同操作。

### 3.3 分支顺序反向的正确性

令 sg 表示停止梯度，构造：

$$
\widetilde J_{\rm direct}
=\sum_d\operatorname{sg}(\mu_\theta(d))C_\theta(d),
$$

$$
\widetilde J_{\rm route,E}
=\sum_d\mu_\theta(d)\operatorname{sg}(C_\theta(d)).
$$

第一项求导：

$$
\nabla\widetilde J_{\rm direct}
=\sum_d\mu_\theta(d)\nabla C_\theta(d).
$$

第二项求导：

$$
\nabla\widetilde J_{\rm route,E}
=\sum_dC_\theta(d)\nabla\mu_\theta(d).
$$

两者相加恰好为 $\nabla J$。这允许逐候选反向、释放激活后再计算控制路径。条件是在两类分支之间不更新参数，并使用同一参数状态下的 μ 和 C。

两个代理的前向值都等于 J，所以代理标量之和为 2J。它的自动微分梯度却等于 $\nabla J$，因为两项各自切断了不同的依赖边。日志必须单独计算真实 J，不能把代理之和当作优化目标的数值。

### 3.4 动态软标签的含义

第 t 次更新生成标签：

$$
s_t=\operatorname{softmax}(-C_{\theta_t}/\tau),\qquad \tau=0.05.
$$

本次反向将 $s_t$ 当作固定目标：

$$
A_t(\theta)
=0.1\,\operatorname{CE}(s_t,\mu_\theta)
+0.01\|\mu_\theta-s_t\|_2^2
+0.1\,\operatorname{KL}(\mu_\theta\Vert\mu_{\rm ref}).
$$

E 在裁剪与优化器处理前的损失梯度为 $\nabla J(\theta_t)+\nabla_\theta A_t(\theta_t)$。J 的条件期望等于该式，实际步骤使用随机路由估计；S 的梯度为 $\sum_d\mu_d\nabla L_d+\nabla A_t$。辅助项不是对复合函数 $A(C_\theta,\mu_\theta)$ 求全导数。停止标签梯度避免模型通过改变“正确标签”来降低辅助损失；它是预先规定的每步监督规则。

S、E、J 使用同样的标签重算与停止梯度规则。J 与 E 的归因因此落在路由梯度估计上。固定参考 $\mu_{\rm ref}$ 来自 warm 读出与零 LoRA 的实际前缀分布，并在训练前保存。

## 4. Jev score 梯度与留一法基线

### 4.1 无偏性

Dirichlet 对数密度为：

$$
\log p_\theta(q)
=\log\Gamma(A)-\sum_d\log\Gamma(\alpha_d)
+\sum_d(\alpha_d-1)\log q_d.
$$

对 $\alpha_j$ 求导：

$$
\frac{\partial\log p(q)}{\partial\alpha_j}
=\psi(A)-\psi(\alpha_j)+\log q_j,
$$

其中 $\psi$ 是 digamma 函数。所有 $\alpha_j>0$，分布支持集不随参数变化，所用线性成本可积，故可交换积分与求导。先将 C 停止梯度：

$$
\begin{aligned}
\nabla_\theta\mathbb E_q[q^\top C]
&=\int(q^\top C)\nabla_\theta p_\theta(q)\,dq\\
&=\int(q^\top C)p_\theta(q)\nabla_\theta\log p_\theta(q)\,dq\\
&=\mathbb E_q[(q^\top C)\nabla_\theta\log p_\theta(q)].
\end{aligned}
$$

它与第 2 节的精确路由梯度相等。若 C 也依赖 θ，则积分求导还有 $\mathbb E_q[q^\top\nabla C_\theta]=\mu^\top\nabla C_\theta$，正是第 3 节的直接生成项。

### 4.2 留一法

采样 $q_1,\ldots,q_M$，M=64，令：

$$
v_m=q_m^\top\operatorname{sg}(C_\theta),\qquad
b_m=\frac1{M-1}\sum_{\ell\ne m}v_\ell.
$$

在固定输入、参数和成本下，$b_m$ 与 $q_m$ 独立。又因为：

$$
\mathbb E_q[\nabla_\theta\log p_\theta(q)]
=\int\nabla_\theta p_\theta(q)\,dq
=\nabla_\theta1=0,
$$

所以：

$$
\mathbb E[b_m\nabla_\theta\log p_\theta(q_m)]
=\mathbb E[b_m]\,
\mathbb E[\nabla_\theta\log p_\theta(q_m)]
=0.
$$

由此得到无偏估计：

$$
\widehat g_J
=\frac1M\sum_{m=1}^M
(v_m-b_m)\nabla_\theta\log p_\theta(q_m).
$$

实现使用正号最小化成本，代理损失为：

$$
\widetilde J_{\rm route,J}
=\frac1M\sum_m\operatorname{sg}(v_m-b_m)
\log p_\theta(\operatorname{sg}(q_m)).
$$

q、v 和 b 在此代理中全部停止梯度。让 q 的梯度继续穿过对数密度，会引入额外密度路径项，使这个 score 估计一般有偏。该项是停止梯度的优势乘以对数密度对 q 的导数，再乘 q 对参数的导数；它不同于线性成本本身的合法路径梯度。主实验只使用 score 估计。

若基线包含当前样本本身，即 $\bar v=M^{-1}\sum_\ell v_\ell$，则：

$$
\mathbb E[(v_m-\bar v)\nabla\log p(q_m)]
=\left(1-\frac1M\right)\nabla J_{\rm route}.
$$

这一缩放来自 $\bar v$ 中的 $v_m/M$。留一基线避免该缩放，并允许 E、J 使用同一学习率及同一梯度系数。

### 4.3 方差与能够检验的结论

令 B 表示当前小批量及其当前参数、四动作成本。对 J 的采样随机性取条件期望：

$$
\mathbb E[\widehat g_J\mid B]=g_E(B).
$$

全协方差公式给出：

$$
\operatorname{Cov}(\widehat g_J)
=\operatorname{Cov}(g_E(B))
+\mathbb E_B[\operatorname{Cov}(\widehat g_J\mid B)].
$$

第二项半正定，所以 J 在相同小批量梯度之外添加采样噪声。精确 E 消除了这一条件采样噪声；并没有消除数据小批量噪声。

这个结论不推出非凸网络训练后 E 必然优于 J。额外噪声能够改变优化轨迹，实际泛化差异由正式 E/J 对照检验。它也不支持“J 多采样就增加了对动作结果的观察”：这里所有动作成本在采样前已经可见。

主实验不裁剪 q、不把线性成本改成门槛 reward，也不使用 argmax(q)。这些操作会改变目标或引入新的偏差，若要研究必须另设实验臂。

## 5. 成本、预算与静态比较项

### 5.1 共享前缀的真实成本

设 $F_{\rm native,\theta}(D)$ 为同检查点关闭控制器后的完整请求成本。它已经包含前缀 D 轮循环、后续 H−1 个输入位置的 D 轮循环、末端和 H 个词表输出。

共享控制额外运行深度 4 的末端前缀。选择 D=4 时该计算与原生路径重合；选择 D>4 时它是额外工作。因此：

$$
F_{\rm plan,\theta}(D)
=F_{\rm native,\theta}(D)
+F_{\rm readout}
+\mathbf1\{D>4\}F_{\rm coda,prefix,\theta}.
$$

控制成本不是“两个末端 token 层”，而是整个已知前缀的两层末端。L=128 时需为 128 个位置计算末端注意力。GPU 调度、缓存管理和内核启动的时间开销另记在端到端时长中。

核心输入位置数为：

$$
N=L+(H-1)=255.
$$

核心 token-step 数为：

$$
N_{\rm core}(D)=ND=255D.
$$

令归一化分母为冻结基础模型深度 16 的固定请求 FLOPs：

$$
F_0=F_{\rm native,base}(16),\qquad
f_\theta(D)=F_{\rm plan,\theta}(D)/F_0.
$$

由于长度、rank 和执行形状固定，f 不随适配器数值变化。采用相同绝对分母使平均 f 与整个评价集的总 FLOPs 约束一致。如果为每个请求单独选不同分母，平均比值就不再等于总成本之比。

### 5.2 指数倾斜校准

给定某个检查点输出的 μ，定义：

$$
Z(b)=\sum_d\mu_d e^{-bf_d},\qquad
\mu_b(d)=\frac{\mu_d e^{-bf_d}}{Z(b)}.
$$

对 b 求导，先计算：

$$
\frac{Z'(b)}{Z(b)}
=-\sum_d\mu_b(d)f_d=-\mathbb E_{\mu_b}[f].
$$

于是：

$$
\frac{d\mu_b(d)}{db}
=\mu_b(d)\left[-f_d+\mathbb E_{\mu_b}[f]\right].
$$

平均预算 $B(b)=\sum_d\mu_b(d)f_d$ 的导数为：

$$
\begin{aligned}
B'(b)
&=\sum_df_d\frac{d\mu_b(d)}{db}\\
&=-\sum_d\mu_b(d)f_d^2
+\left(\sum_d\mu_b(d)f_d\right)^2\\
&=-\operatorname{Var}_{\mu_b}(f)\le0.
\end{aligned}
$$

逐请求求均值后导数仍然非正，故校准集预算可用二分法求解。只要至少有两个不同成本且其概率为正，有限 b 区间上导数严格为负。

在 b 趋于正无穷和负无穷时，分布分别集中于最小与最大成本动作。预算 1 必须处于这两个极限之间。该检查应使用包含控制开销的实际成本账本。

### 5.3 相同模型的静态策略

给定同一个 θ，关闭控制器的静态成本为 $f_{{\rm static},\theta}(d)=F_{{\rm native},\theta}(d)/F_0$，它没有路由读出或额外前缀末端开销。以下简记为 $f_{{\rm static},d}$。在校准集上计算四个固定深度的均值 $(f_{{\rm static},d},\bar L_{\theta,d})$，输入无关混合求解：

$$
\min_{\pi\ge0}\sum_d\pi_d\bar L_{\theta,d},
\qquad
\sum_d\pi_df_{{\rm static},d}\le1,\qquad
\sum_d\pi_d=1.
$$

该策略按请求抽一个深度，质量是各动作 NLL 的期望。它不等于词表概率混合的 NLL：

$$
\sum_d\pi_d[-\log P_d(y)]
\ne-\log\left[\sum_d\pi_dP_d(y)\right].
$$

右侧需要同时获得多个模型的概率，执行成本不同。本协议评价左侧。

线性规划至少有一个极点最优解。预算约束激活时，两个独立等式限制支持权重。若某极点有三个以上正权重，则存在非零方向 v 使 $\sum_dv_d=0$ 且 $\sum_df_{{\rm static},d}v_d=0$；沿正负足够小的 v 移动仍可行，原点就是两个可行点的中点，违背极点定义。因此最优解可取至多两个支持点。预算未激活时，可取单一最优点。

对 E 自己的检查点做这个比较，固定了生成器质量，仅移除输入相关路由。主比较池再加入普通末端适配和冻结基础模型，使动态方法同时面对表示适配与静态分配两个解释。

## 6. 资源数目的逐项推导

### 6.1 LoRA 参数

一个 $d_{\rm in}\to d_{\rm out}$ 线性层的 rank-r LoRA 写为：

$$
\Delta W=sBA,\qquad
A\in\mathbb R^{r\times d_{\rm in}},\quad
B\in\mathbb R^{d_{\rm out}\times r}.
$$

其参数数为 $r(d_{\rm in}+d_{\rm out})$。每个末端块使用四个实际融合矩阵：

| 矩阵 | 输入→输出 | LoRA 参数 |
|---|---|---:|
| fused QKV | h→3h | 4rh |
| 注意力输出 | h→h | 2rh |
| fused gate/up | h→2m | r(h+2m) |
| 前馈 down | m→h | r(m+h) |

因此两层总数为：

$$
\begin{aligned}
P_{\rm LoRA}
&=2r[4h+2h+(h+2m)+(m+h)]\\
&=2r(8h+3m)\\
&=2\cdot8[8\cdot5280+3\cdot17920]\\
&=1,536,000.
\end{aligned}
$$

按 FP32 参数、梯度、一阶和二阶 Adam 状态四份存储：

$$
M_{\rm optim}=4\cdot4P_{\rm LoRA}
=24,576,000\ {\rm bytes}
=23.4375\ {\rm MiB}.
$$

额外 BF16 参数转换副本、临时梯度、激活和框架工作区另计。这一数字不能代替峰值显存测量。

若 LoRA 两矩阵都初始化为零，则：

$$
\frac{\partial(BA)}{\partial A}\propto B=0,\qquad
\frac{\partial(BA)}{\partial B}\propto A=0.
$$

首步两者均无梯度。A 随机、B 为零时，B 的梯度能启动适配，更新 B 后 A 才开始接收梯度。

### 6.2 KV 和特征缓存

深度 D 的展开层数为：

$$
n_{\rm layer}(D)=2+4D+2.
$$

每层的 key 和 value 都有 55 个头、每头 96 维，合计 h=5,280。BF16、batch 1、T 个位置的完整 KV 容量为：

$$
M_{\rm KV}(D,T)=2_{\rm K,V}\cdot2_{\rm bytes}\cdot5280
\cdot(2+4D+2)\cdot T.
$$

D=32、T=255 时约 0.662 GiB；若错误地按 4,096 个位置为每个展开层预分配，则约 10.635 GiB。后者在 16 GB 卡上会挤掉模型与激活所需空间。

每文档保存四个深度的 255 位置末端输入，以及独立 128 位置控制前缀：

$$
M_{\rm feature/doc}
=2\cdot5280(4\cdot255+128)
=12,122,880\ {\rm bytes}.
$$

最大 5,664 篇的纯特征总量为：

$$
M_{\rm feature,total}
=5664\cdot12,122,880
=68,663,992,320\ {\rm bytes}
\approx63.9483\ {\rm GiB}.
$$

清单、索引、校验、临时文件以及额外初始化复核缓存占用另计，所以协议预留 80 GiB。

## 7. 统计判断的边界

NLL 的配对差先按文档构造，再平均三个训练实例。bootstrap 重采样文档，因此区间估计的是这三个固定实例在同分布新文档上的平均差；训练实例总体的方差没有被三次训练充分刻画。

设单文档差标准差为 σ，真实规划效应为 δ，则平均差在正态近似下满足：

$$
\bar g\approx\mathcal N(\delta,\sigma^2/n).
$$

单侧水平 α 下，拒绝零效应要求：

$$
\bar g>z_{1-\alpha}\frac{\sigma}{\sqrt n}.
$$

若功效为 $1-\gamma$，则：

$$
\begin{aligned}
\Pr\left(\bar g>
z_{1-\alpha}\frac{\sigma}{\sqrt n}\right)&\ge1-\gamma\\
\frac{\delta\sqrt n}{\sigma}-z_{1-\alpha}&\ge z_{1-\gamma}\\
n&\ge
\left[\frac{(z_{1-\alpha}+z_{1-\gamma})\sigma}{\delta}\right]^2.
\end{aligned}
$$

要求点估计同时达到 δmin，则还需：

$$
\Pr(\bar g\ge\delta_{\min})\ge1-\gamma
\quad\Longleftarrow\quad
n\ge\left[\frac{z_{1-\gamma}\sigma}
{\delta-\delta_{\min}}\right]^2.
$$

对同一个均值统计量，两项通过事件的交集等价于超过两个阈值中较大者，所以取两项所需 n 的最大值。主计划以 α=0.025、$1-\gamma=0.8$ 规划每项检验，再用 Holm 校正两项实际 p 值。

例如 Jev 配对差在筛选集上的标准差为 0.02，预定放大 1.5 倍得到 0.03。以 δ=0.002 规划，拒绝零效应约需 1,766 篇，上取到 1,792 篇。若原标准差为 0.03，规划数量约 3,974 篇，超过 2,048 的上限；此时应报告预算内功效不足。

显著且点估计超过 0.001，不等于证明总体均值至少 0.001。后者要直接检验非零零假设，并在样本量公式中把 δ 换成 $\delta-\delta_{\min}$。本协议不混用这两种结论。

## 8. 已完成的数值检查与待执行检查

[CPU 检查脚本](../scripts/check_experiment_math.py)使用 Python 3.12.13、PyTorch 2.8.0、CPU float64 和固定种子 20261011。其[实际结果文件](EXPERIMENT_MATH_CHECK.v1.json)同时保存软件版本、运行时间和脚本 SHA256。检查使用小张量，不加载语言模型。

| 检查 | 结果 |
|---|---|
| 解析均值、解析梯度与有限差分 | 通过；梯度最大绝对误差约 5.83×10⁻¹² |
| 4,096 组、每组 64 个 Dirichlet 样本的留一梯度 | 通过；最大偏差约 2.75 个蒙特卡洛标准误 |
| 常数成本和平移不变性 | 通过 |
| 生成梯度与路由梯度的联合分解 | 通过；标量例子 −1−0.25=−1.25 |
| 指数倾斜预算导数 | 通过；等于 −Var(f) |
| LoRA 初始化、参数、KV 与特征容量 | 通过 |

蒙特卡洛检查验证本次固定输入下的梯度实现，解析推导给出一般无偏性。真实 Huginn 的前缀因果性、续算缓存、共享参数更新、内核数值误差、显存与吞吐必须在 RTX 5080 先导中实测。当前文件没有提供这些真实模型结果。

固定源码依据是 [Huginn 实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)：full 缓存默认在第 149 行；融合投影与因果注意力在第 449–502 行；末端与归一化在第 692–717 行；循环与原始状态初始化在第 721–810 行；单步续算从第 839 行开始。所有行号针对固定 revision，运行时另保存本地源码哈希。
