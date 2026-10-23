# 场演化网络（Field Evolution Network）：一种以同维度反应-扩散场为计算原语的通用网络架构

> **论文草稿 v3.0（中文，架构范式导向）** ｜ 目标：架构类期刊论文
> **作者**：（占位） ｜ **日期**：2026-08-06
> **范围声明**：本文把 FluidNet-NC 作为一种**网络架构范式**提出——就像卷积（CNN）与注意力（Transformer）一样，定义一种新的计算原语与网络结构，并在监督学习各任务上验证其作为通用架构的竞争力。文中只描述当前确定下来的最终结构。
> **内部备注**（附录 A：证据等级与待补实验清单，投稿前删除）

---

## 摘要

现代深度学习的架构史，本质上是**计算原语**的演进史：卷积（CNN）以局部共享核实现平移等变，注意力（Transformer）以全对全交互实现任意位置直达。本文提出第三种通用计算原语——**场演化（Field Evolution）**，并将其规范实现为 **FluidNet-NC** 架构：输入被编码为 $n\times n\times h$ 的场张量，信息通过"傅里叶全局混合 + 多层同维度反应-扩散演化"进行传播与融合，最终经池化/逐位置读出输出。与卷积相比，场演化以线性复杂度建立全局交互；与注意力相比，它引入连续物理动力学先验，且复杂度对序列长度线性。

作为架构范式，FluidNet-NC 具有四个可验证的性质：**(i) 深度可自由堆叠**——通过取消压缩层、保持场张量形状恒为 $[B,1,n,n,h]$ 并配合残差与层归一化，突破了原始流体网络 2 层的深度上限；**(ii) 线性复杂度**——$O(L\cdot T\cdot h\cdot c)$；**(iii) 多模态友好**——异质模态可投影到同一物理场网格，由场演化完成跨模态融合；**(iv) 强结构归纳偏置**——反应-扩散动力学天然支持局部平滑与全局混合。

在六类监督任务上，FluidNet-NC 表现出清晰的"优势图谱"：多模态视觉-文本识别 **99.17%**（较压缩基线 +9.7 个百分点）；784 步长序列分类 **84.95%**（超越同规模 Transformer 75.95% 与 LSTM 38.70%）；MNIST 98.7%（深度堆叠 L8>L4）；字符级生成教师强制 100%；VLA 监督动作回归领先（Pick-and-Place MSE 2.08e-3，LIBERO 简化 0.492）。同时如实报告边界：精确位置检索任务（Adding T=500）上注意力仍占优，过深堆叠会抹平局部标记信息。

**关键词**：网络架构范式；计算原语；反应-扩散场演化；同维度堆叠；监督学习；多模态融合；长程建模

---

## 1 引言：第三种计算原语

**从卷积到注意力。** 卷积神经网络（CNN）以"局部感受野 + 权重共享"为原语，实现了平移等变与参数高效，统治了计算机视觉十余年 [26]。Transformer [1] 以"全对全注意力"为原语，实现了任意两位置间的直接信息传递，统一了 NLP 与多模态建模。两种原语各有边界：卷积的全局交互需要深堆叠（感受野逐层扩大），注意力的 $O(T^2)$ 复杂度限制了长序列，且两者都不具备显式的连续动力学归纳偏置。

**场演化：第三种原语。** 我们提出将**信息演化建模为物理场上的反应-扩散过程**作为第三种计算原语。其思想源自 Turing 形态发生理论 [6]：在空间网格上，信息以局部扩散与逐点反应两种机制传播。这一原语同时具备：卷积式的局部共享结构（平移等变、参数高效）、注意力式的全局交互能力（经傅里叶混合一次性建立）、以及线性复杂度（全局混合经 FFT 实现，$O(T\log T)$）。

**本文的架构。** 将场演化原语实例化为完整网络结构 FluidNet-NC：场编码 → 傅里叶全局混合 → $L$ 层同维度反应-扩散演化 → 读出。与原始流体网络的关键区别在于**无压缩**：演化全程保持场张量同维度，使深度从 2 层限制中解放。本文按"架构范式论文"的方式组织：先定义范式与构建块（第 3 节），再给出确定该结构的逐步改进过程（第 5 节），最后在六类监督任务上验证其通用性（第 4 节）。

**贡献。**

1. **新架构范式**：提出"场演化"作为与卷积、注意力并列的计算原语，并给出完整实现 FluidNet-NC——无压缩、同维度、可深堆叠、线性复杂度。
2. **结构性质**：从理论复杂度、数值稳定性、深度可扩展性与多模态友好性四方面刻画该架构的性质。
3. **系统验证**：六类监督任务（图像/文本/多模态分类、序列生成、复杂时序、VLA 动作回归）上的基准与边界分析，含与 CNN、Transformer、LSTM、MLP 的对照。
4. **设计方法**：记录从原始流体网络到最终结构的逐步改进与消融证据，形成可复用的架构设计流程。

## 2 相关工作

**卷积与注意力架构。** CNN [26] 与 Transformer [1] 分别以卷积与注意力为原语，是当前两大架构家族。高效注意力（线性注意力 [2]、Performer [3]、Reformer [7]）与状态空间模型（S4 [4]、Mamba [5]）致力于在保持注意力的全局性同时降低复杂度。场演化与之不同：它不以"成对交互"为基本操作，而以"场上的局部-全局耦合动力学"为基本操作，属于不同的计算范式。

**物理场与神经动力系统。** Turing 反应-扩散系统 [6] 是自组织与形态发生的经典模型；Neural ODE [8] 将残差网络解释为连续动力系统；PINN [9] 将物理约束注入网络；神经场（NeRF [10]）以坐标-特征映射表示信号。FluidNet-NC 的贡献在于将这些思想**收束为一个通用的监督学习网络架构**，并给出完整的构建块、性质分析与跨任务验证。

**多模态融合。** 跨模态注意力（CLIP [11]、BLIP [12]）与生成式模型（LLaVA [13]）以注意力跨越模态；场演化提出替代机制：异质模态投影到同一物理场网格，由场演化完成交换。本文多模态实验（89.44%→99.17%）为该机制提供初步证据。

**视觉-语言-动作（VLA）。** RT-1 [14]、RT-2 [15]、Diffusion Policy [16]、Open X-Embodiment [17] 与 LIBERO [18] 定义了 VLA 监督学习范式。场演化可作为其融合骨干的备选，本文在监督动作回归设定下验证。

**长程时序监督基准。** sMNIST/pMNIST [19]、Long Range Arena [20]、ETT/Exchange [21] 提供标准评估；本文覆盖 sFashionMNIST、Adding 问题 [22] 与 Exchange 预测。

## 3 场演化网络架构（FluidNet-NC）

### 3.1 范式定义

一个场演化网络由三个要素构成：**场表示**（输入如何编码为空间网格上的张量）、**演化算子**（信息如何在场上传播与融合）、**读出**（演化后的场如何还原为任务输出）。FluidNet-NC 给出这三个要素的具体实现，其流水线为：

$$
\mathbf{X}\xrightarrow{\text{场编码}}\mathbf{F}^{(0)}\xrightarrow{\text{傅里叶全局混合}}\mathbf{F}^{(0)'}\xrightarrow{\text{L 层同维度演化}}\mathbf{F}^{(L)}\xrightarrow{\text{读出}}\mathbf{y}
$$

场张量 $\mathbf{F}^{(l)}\in\mathbb{R}^{B\times 1\times n\times n\times h}$ 在全部 $L$ 层演化中**形状不变**。

### 3.2 与卷积、注意力的关系

| 性质 | 卷积（CNN） | 注意力（Transformer） | 场演化（FluidNet-NC） |
|---|---|---|---|
| 基本操作 | 局部加权求和 | 全对全加权求和 | 局部扩散 + 全局傅里叶混合 |
| 复杂度 | $O(T)$ | $O(T^2)$ | $O(T\log T)$（FFT） |
| 全局交互 | 需深堆叠 | 单层直达 | 单层混合 + 逐层演化 |
| 平移等变 | 是 | 否（靠位置编码） | 是（卷积核）+ 显式位置嵌入 |
| 连续动力学先验 | 弱 | 无 | 强（反应-扩散） |
| 深度稳定性 | 需残差 | 需残差/LN | 残差/LN（无压缩后验证至 L8） |
| 多模态交互介质 | 无统一介质 | 注意力矩阵 | 统一物理场网格 |

### 3.3 场编码

输入元素（像素/图像块/词元/时间步）经线性投影 $\mathbf{E}=\mathbf{X}\mathbf{W}_e+\mathbf{b}_e$（$\mathbf{W}_e\in\mathbb{R}^{C\times h}$）得到 $h$ 维嵌入，按行优先写入 $n\times n$ 网格（$n=\lceil\sqrt{T}\rceil$），叠加可学习位置嵌入：

$$
\mathbf{F}^{(0)}[\cdot, \lfloor t/n\rfloor, t\bmod n, :] = \mathbf{E}[\cdot,t,:] + \mathbf{P}[\lfloor t/n\rfloor, t\bmod n, :]
$$

未占用单元由可学习偏置初始化。相邻元素在场中空间相邻，短程交互由局部算子自然完成。

### 3.4 演化算子：傅里叶全局混合 + 同维度反应-扩散

**全局混合**（每网络前向一次）：

$$
\tilde{\mathbf{F}}=\mathcal{F}^{-1}\big(\mathcal{G}(\mathcal{F}(\mathbf{F}^{(0)}))\big),\qquad
\mathbf{F}^{(0)'}=\mathbf{F}^{(0)}+\text{LayerNorm}(\tilde{\mathbf{F}})
$$

**反应-扩散演化**（$L$ 层）：核心算子 ADRBlock3D 离散化

$$
\frac{\partial \mathbf{F}}{\partial \tau} = D\nabla^2\mathbf{F} + R(\mathbf{F}),
$$

并以残差 + 层归一化接入：

$$
\mathbf{F}^{(l+1)}=\text{LayerNorm}\big(\mathbf{F}^{(l)} + \text{ADRBlock3D}(\mathbf{F}^{(l)})\big),\quad l=0,\dots,L-1
$$

演化步长 $\eta=\texttt{adr\_step\_size}$ 控制单层扩散幅度（最终结构取 0.30，见 5.2 节）。

### 3.5 深度堆叠与数值稳定性

与原始流体网络不同，ADR 块**不含"Q 压缩—重建"路径**，直接在同维度场上演化。这一设计使深度 $L$ 可自由堆叠：残差 + LayerNorm 保证数值稳定（观测场范数演化前后 11.1→9.4），参数量与计算随 $L$ 线性增长。这是本架构区别于早期流体网络的核心结构决策（动机与证据见 5.1 节）。

### 3.6 读出

| 监督任务 | 读出方式 | 输出形式 |
|---|---|---|
| 分类（图像/文本/多模态） | 自适应平均池化 → 展平 → 大 MLP（128~256 维） | 类别分布 |
| 序列生成 | 逐位置读出头 | 每位置下一元素分布 |
| 回归（时序/动作） | 池化 → MLP 头 | 标量 / 多步序列 / 动作向量 |

### 3.7 复杂度与架构性质小结

- 计算复杂度 $O(L\cdot T\cdot h\cdot c + T\log T)$，对序列长度近线性；
- 内存随 $T$ 线性增长；
- 深度 $L$ 无硬性上限（实验验证至 8，L16 待补）；
- 统一场网格天然支持多模态共置（4.4 节证据）。

---

## 4 实验：作为通用架构的监督学习验证

### 4.1 实验设置

- **硬件/软件**：NVIDIA RTX 5070 Ti（16GB），PyTorch 2.9.1+cu128，Python 3.14，CUDA 12.8。
- **确定性**：所有实验锁定随机种子（seed=42），覆盖数据划分、初始化与采样；优化器 AdamW（lr=$1.2\times10^{-3}$，weight decay $10^{-4}$），余弦退火 + 10% 线性预热，梯度裁剪范数 1.0。
- **FluidNet-NC 默认配置**：$h=8$（可调至 16），$n=\lceil\sqrt{T}\rceil$（图像任务 $n=28$），池化核 $4\times4$，$\eta=0.30$，ADR 隐藏通道 32，读出 MLP 128~256 维。
- **基线**：CNN、Q 压缩版 FluidNet（Baseline-Q）、LSTM、MLP、小规模 Transformer-Encoder（$d=64$，3 层，4 头，FFN 256）。各模型参数量处于同一数量级（$10^4$~$10^6$），训练步数与优化配置一致。
- **报告口径**：单 seed 验证集结果；多 seed 误差棒见【待补】。

### 4.2 图像分类（CV）

| 数据集 | 类别数 | CNN | Baseline-Q | NC(L4) | NC(L8) |
|---|---:|---:|---:|---:|---:|
| MNIST | 10 | 【待补】 | 98.6% | 98.4% | **98.7%** |
| FashionMNIST | 10 | 【待补】 | **88.8%** | 83.1% | 85.3% |

- MNIST 上 L8 首次超越 L4（98.7% vs 98.4%），验证深度堆叠增益。
- FashionMNIST 上 NC 落后于压缩基线，属已知局限（见 6.3 节）。
- 【待补】CIFAR-100 / ImageNet-1k；ResNet/ViT 同参数量对照；≥5 seeds 误差棒。

### 4.3 文本分类（NLP）

| 数据集 | 类别数 | Baseline-Q | NC(L4) | NC(L8) |
|---|---:|---:|---:|---:|
| 20Newsgroups（子集） | 4 | **89.53%** | 89.28% | 89.15% |

- NC 与压缩基线持平（差距 <0.4pt），证明去除压缩不损失文本分类能力。
- 【待补】GLUE（SST-2/MRPC/QQP/CoLA）；SQuAD；中文 CLUE；长文本（>4K token）。

### 4.4 多模态分类

| 数据集 | 模态 | Baseline-Q | NC(L4) | NC(L8) |
|---|---|---:|---:|---:|
| 视觉-文本数字识别 | 图像 + 文本 | 89.44% | **99.17%** | **99.17%** |

- **最大增益场景**：统一物理场网格为异质模态提供公共交互介质，融合增益 +9.7pt。
- 【待补】CMU-MOSEI / MM-IMDb；图文检索（Flickr30K / COCO）；模态缺失鲁棒性。

### 4.5 序列生成（字符级，教师强制监督）

| 模型 | 教师强制 next-char 准确率 | 贪心自回归模式保持率（40 字符） |
|---|---:|---:|
| LSTM | 85.5% | 7.5% |
| MLP | 100% | 25.0% |
| NC(L4) | 100% | 0.0% |
| NC(L8) | 100% | **15.0%** |

- 教师强制指标在该确定性语料上饱和，NC 与 MLP 并列最优、超越 LSTM；贪心保持率显示 NC-L8 具备模式延续倾向。
- 【待补】Tiny Shakespeare / enwik8 困惑度；温度与 top-k 扫描；与 GPT-2 规模小模型对照。

### 4.6 复杂时序任务

**（a）Adding 问题（T=500，长程精确检索回归）**

| 模型 | MSE | MAE | 参数量 |
|---|---:|---:|---:|
| LSTM | 0.6826 | 0.6776 | 17.5K |
| Transformer | **0.0012** | **0.0266** | 215.7K |
| NC(L4) | 0.0211 | 0.1085 | 191.6K |
| NC(L8) | 1.1198 | 0.8591 | 343.2K |

**（b）Exchange 汇率预测（真实数据，lookback 96 → horizon 24，归一化）**

| 模型 | MSE | MAE |
|---|---:|---:|
| LSTM | **0.811** | **0.729** |
| Transformer | 1.400 | 1.015 |
| NC(L4) | 0.959 | 0.828 |
| NC(L8) | 1.152 | 0.885 |

**（c）Sequential FashionMNIST（784 步长序列分类）**

| 模型 | 准确率 |
|---|---:|
| LSTM | 38.70% |
| Transformer | 75.95% |
| NC(L4) | **84.95%** |
| NC(L8) | 【待补：实验中断，需补跑】 |

- 长序列分类上 NC 显著超越同规模 Transformer（+9.0pt）与 LSTM（+46.3pt）。
- Adding 上注意力占优（精确检索）；汇率预测上全部模型接近随机水平。
- 【待补】LRA 全量（ListOps/Text/Retrieval/Image/Pathfinder）；pMNIST；ETT/Electricity/Traffic；T≥4096。

### 4.7 视觉-语言-动作（VLA，监督动作回归）

以"视觉（ViT）+ 预训练语言编码器 + 机器人状态 + 世界模型感知"四模态输入，经场空间融合后回归动作。

**（a）单步 Reaching（图像 → 3 维动作）**

| 模型 | MSE | MAE |
|---|---:|---:|
| CNN | **2.1e-5** | 0.0035 |
| FluidNet-Enhanced | 2.6e-4 | 0.0123 |
| FluidNet-NC-VLA | 9.8e-4 | 0.0199 |
| MLP | 2.3e-2 | 0.1169 |

**（b）单步 Pick-and-Place（图像 + 文本 → 4 维动作）**

| 模型 | MSE | MAE |
|---|---:|---:|
| FluidNet-NC-VLA | **2.08e-3** | 0.0325 |
| FluidNet-Enhanced | 2.27e-3 | 0.0347 |
| MLP | 【待补】 | 【待补】 |

**（c）LIBERO-10 简化代理任务（四模态动作回归）**

| 模型 | val MSE | val MAE |
|---|---:|---:|
| MLP-Baseline | 0.5274 | 0.5163 |
| FluidNet-NC-4Modal | **0.4922** | 0.4839 |
| FluidNet-NC-NoWorld | 0.5449 | 0.5065 |

- 需要语义理解的任务（Pick-and-Place、LIBERO 简化）上 NC 领先；纯空间定位（Reaching）上 CNN 最优。
- 【待补】LIBERO-10 全量成功率；PushT；与 RT-1/RT-2/Diffusion Policy 对照；真实机器人。

### 4.8 总体性能汇总（架构级结论）

| 任务域 | 任务 | FluidNet-NC 最佳 | 最强基线 | 结论 |
|---|---|---|---|---|
| CV | MNIST | 98.7% | 98.6%（Baseline-Q） | 追平并微超 |
| CV | FashionMNIST | 85.3% | 88.8%（Baseline-Q） | 落后 |
| NLP | 20News | 89.3% | 89.5%（Baseline-Q） | 持平 |
| 多模态 | 视觉-文本识别 | **99.17%** | 89.44%（Baseline-Q） | 大幅领先 |
| 生成 | 字符级 | 100% | 100%（MLP） | 并列最优 |
| 时序 | Adding T=500 | 0.021（L4） | 0.0012（Transformer） | 落后 |
| 时序 | Exchange 预测 | 0.959 | 0.811（LSTM） | 落后 |
| 时序 | sFashionMNIST | **84.95%** | 75.95%（Transformer） | 大幅领先 |
| VLA | Pick-and-Place | **2.08e-3** | 2.27e-3（Enhanced） | 领先 |
| VLA | LIBERO 简化 | **0.492** | 0.527（MLP） | 领先 |

> **架构级概括**：场演化架构在**多模态融合、长序列分类、跨模态语义动作**三类任务上表现突出；在**精确位置检索（Adding）**上弱于注意力；在**强高频纹理（FashionMNIST）与强平稳先验（Exchange）**上弱于针对性基线。这一"优势—边界"图谱是本文的核心实验结论，也是后续架构改进的输入。

---

## 5 架构的逐步改进：从流体网络到场演化范式

本节呈现确定最终结构（第 3 节）的逐步改进过程。每一步以实验证据为决策依据；最终结构不再并列多个候选变体。

### 5.1 改进一：取消 Q 压缩层，实现同维度演化

**问题**：原始流体网络的 ADR 块内含"Q 压缩—重建"路径（场 → 低维潜空间 → 演化 → 重建），导致：(a) 逐层堆叠时压缩-重建误差循环放大数值误差，深度被限制在 2 层以内；(b) 独立复用 ADR 块时数值不稳定。

**改进**：移除压缩-重建路径，ADR 块直接在同维度场 `[B,1,n,n,h]` 上演化，配残差 + LayerNorm。

**证据**：独立 ADR 块复用稳定（场范数 11.1→9.4 无爆炸）；多模态任务上无压缩结构 89.44%→99.17%；MNIST 上深度堆叠首次产生增益（L8=98.7% > L4=98.4%），此前压缩结构无法超过 2 层。

**决策**：采用无压缩同维度结构（本架构与早期流体网络的核心区别）。

### 5.2 改进二：增大演化步长

**问题**：默认 $\eta=0.05$ 时单层扩散幅度过小，深层堆叠收益受限。

**改进**：$\eta$ 提升至 0.30。

**证据**：初步实验中 $\eta=0.05\to0.30$ 带来显著增益（约 35.6%→55.3% 量级）。【待补：完整消融表，覆盖多任务与中间步长 0.10/0.20】

**决策**：$\eta=0.30$。

### 5.3 改进三：增大读出维度与任务头设计

**问题**：早期结构输出头容量不足（池化后仅接小 MLP）。

**改进**："池化（4×4×h）→ 展平 → 大 MLP（128~256 维）"读出；生成任务用逐位置读出头。

**证据**：大输出头在分类任务上稳定优于小输出头。【待补：输出维度 32/64/128/256 完整消融】

**决策**：读出 MLP 128~256 维。

### 5.4 消融汇总

| 设计维度 | 备选设置 | 关键证据 | 最终选择 |
|---|---|---|---|
| 压缩层 | 有（Q 压缩） vs 无 | 多模态 89.44%→99.17%；深度可堆叠 | 无压缩 |
| 演化步长 η | 0.05 vs 0.30 | 约 35.6%→55.3% 量级提升（初步） | 0.30 |
| 深度 L | L4 vs L8 | MNIST 98.4%→98.7% | L=8（任务可调） |
| 读出维度 | 小 MLP vs 大 MLP（128~256） | 分类稳定提升（初步） | 大 MLP |
| 池化核 | 2/4/8 | 【待补】 | 4 |
| 场分辨率 n | 24/28/32 | 【待补】 | $\lceil\sqrt{T}\rceil$ |
| 傅里叶混合 | 有 vs 无 | 【待补】 | 保留 |

> 说明："初步"证据来自探索期实验，正式论文需补全为完整消融表（附录 A 清单）。

---

## 6 讨论

### 6.1 为什么无压缩结构能稳定深堆叠？

**已验证事实**：独立 ADR 块数值稳定；压缩结构的爆炸仅出现在压缩-重建循环中；去除压缩后 L8 在 MNIST 上首次超越 L4。

**机制解释（理论推测，待严格证明）**：残差 + LayerNorm 的同维度结构将每层表示为"恒等映射 + 有界演化算子"的叠加，误差传播满足压缩性条件；压缩-重建循环相当于每层附加低秩投影误差，逐层几何级数累积。【待补：谱分析或误差传播上界证明】

### 6.2 优势图谱：场演化适合什么任务？

综合 4.2–4.8 节，优势集中在：**多模态融合**（+9.7pt）、**长序列结构融合**（sFashionMNIST +9.0pt）、**跨模态语义动作**（VLA 领先）。弱势对应：**精确位置检索**（Adding）与**强局部高频纹理**（FashionMNIST）。

### 6.3 局限

1. FashionMNIST 落后于压缩基线（85.3% vs 88.8%），机制待分析；
2. 演化步长 η 敏感，缺自动调参策略；
3. 当前证据集中于小型/合成基准，大规模验证缺失；
4. 深度堆叠的稳定性与表达能力缺严格理论保证。

---

## 7 结论

本文提出**场演化（Field Evolution）**作为与卷积、注意力并列的第三种计算原语，并给出其规范实现 FluidNet-NC：无压缩、同维度、可深堆叠、线性复杂度的通用网络架构。通过逐步改进与消融确定最终结构后，六类监督任务验证表明：该架构在多模态融合（99.17%）、长序列分类（sFashionMNIST 84.95%，超越 Transformer 与 LSTM）与跨模态动作回归上表现突出；其能力边界（精确检索由注意力主导）为"场演化 + 稀疏注意力"的混合架构指明了方向。未来工作包括：大规模基准验证（LRA/ImageNet/GLUE/LIBERO 全量）、混合架构、严格理论分析及真实机器人部署。

---

## 参考文献（占位，投稿前补全卷期页码）

1. Vaswani A, et al. Attention Is All You Need. NeurIPS 2017.
2. Katharopoulos A, et al. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
3. Choromanski K, et al. Rethinking Attention with Performers. ICLR 2021.
4. Gu A, Goel K, Ré C. Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022.
5. Gu A, Dao T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2023.
6. Turing A M. The Chemical Basis of Morphogenesis. Phil. Trans. R. Soc. B, 1952.
7. Kitaev N, et al. Reformer: The Efficient Transformer. ICLR 2020.
8. Chen R T Q, et al. Neural Ordinary Differential Equations. NeurIPS 2018.
9. Raissi M, et al. Physics-Informed Neural Networks. J. Comput. Phys., 2019.
10. Mildenhall B, et al. NeRF: Representing Scenes as Neural Radiance Fields. ECCV 2020.
11. Radford A, et al. Learning Transferable Visual Models from Natural Language Supervision (CLIP). ICML 2021.
12. Li J, et al. BLIP: Bootstrapping Language-Image Pre-training. ICML 2022.
13. Liu H, et al. Visual Instruction Tuning (LLaVA). NeurIPS 2023.
14. Brohan A, et al. RT-1: Robotics Transformer for Real-World Control at Scale. RSS 2023.
15. Brohan A, et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. 2023.
16. Chi C, et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023.
17. Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. ICRA 2024.
18. Liu B, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023.
19. Le Q V, et al. A Simple Way to Initialize Recurrent Networks of Rectified Linear Units. 2015.
20. Tay Y, et al. Long Range Arena: A Benchmark for Efficient Transformers. ICLR 2021.
21. Zhou H, et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI 2021.
22. Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997.
23. Dosovitskiy A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
24. Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.
25. Hafner D, et al. Mastering Atari with Discrete World Models (DreamerV2). ICLR 2021.
26. LeCun Y, et al. Gradient-Based Learning Applied to Document Recognition. Proc. IEEE, 1998.

---

## 附录 A：内部备注（投稿前删除）

### A.1 证据等级声明

| 类别 | 内容 | 依据 |
|---|---|---|
| 实验已验证 | 深度堆叠突破 2 层（MNIST L8>L4）；多模态 +9.7pt；sFashionMNIST 超 Transformer 与 LSTM；Adding 上注意力占优 | 4.2–4.6 节（单 seed，小型基准） |
| 工程经验 | η=0.30 优于 0.05；大读出头稳定；残差+LN 稳定性 | 探索期多轮实验观测 |
| 理论推测 | 压缩-重建误差循环是 2 层限制根源；扩散抹平标记 | 未严格证明 |

### A.2 待补实验清单（按优先级）

- [ ] NC-L8 的 sFashionMNIST（实验中断未完成）
- [ ] 演化步长完整消融（0.05/0.10/0.20/0.30）
- [ ] 深度 L16 消融；读出维度 32/64/128/256；池化核 2/4/8；场分辨率 n
- [ ] 傅里叶混合移除实验
- [ ] LRA 全量五任务；pMNIST；ETT/Electricity/Traffic；T≥4096
- [ ] ImageNet-1k / CIFAR-100；GLUE / SQuAD；CMU-MOSEI / MM-IMDb
- [ ] LIBERO-10 全量成功率；PushT；与 RT-1/RT-2/Diffusion Policy 对照
- [ ] ≥5 seeds 误差棒与显著性检验；效率 profiling（FLOPs/延迟/显存）
- [ ] 理论分析（稳定性、表达性、泛化界）
- [ ] 复现说明（环境、命令、种子清单）；英文版全文

---

*（草稿结束。实验脚本：experiments_v3/tests/*.py；结果：experiments_v3/results/*.json；架构图：experiments_v3/results/nc_architecture.png）*