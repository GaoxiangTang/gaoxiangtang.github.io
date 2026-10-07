---
title: "Parallel two-qubit gates in ion traps and their application to QEC"
date: 2026-10-07 00:00:00 +0800
layout: post
lang: zh
translation_key: parallel-two-qubit-gates-in-ion-traps
slug: parallel-two-qubit-gates-in-ion-traps
permalink: /zh/blog/2026/parallel-two-qubit-gates-in-ion-traps/
---

本 blog 介绍我们的论文 [tang2026](https://arxiv.org/abs/2609.04081)，主要回答以下三个问题：

1. 为什么离子阱上的量子纠错需要并行两比特门？
2. 如何实现低 crosstalk、对噪声鲁棒、功率要求低且波形优化简单的并行两比特门？
3. 这些并行门能用于什么？

## 为什么离子阱上的量子纠错需要并行两比特门？

考虑 BB $$[248,10,18]$$ code [cain2026](https://arxiv.org/html/2603.28627v1#A2)，它用 248 个物理量子比特编码 10 个逻辑量子比特，每个 cycle 需要 1488 个两比特门。若两比特门时间取 $$200\,\mu\mathrm{s}$$ [hughes2025](https://arxiv.org/html/2510.17286v1#S3.SS1)，串行执行每个 cycle 需要 $$297.6\,\mathrm{ms}$$。取 $$T_2\approx3\,\mathrm{s}$$ [egan2021](https://arxiv.org/pdf/2009.11482#page=16)，按指数退相干模型估算，每个 cycle 平均会产生约 12 个由 idle dephasing 导致的 Z flip，超过 distance 18 下保证可纠正的 8 个错误。

进一步，对于 distance 18 的 BB248，在低错误率下，由 idle dephasing 引起的每个 cycle 的逻辑错误率满足 $$\epsilon_{\mathrm{cyc}}\sim t_{\mathrm{cyc}}^{\lfloor d/2\rfloor}=t_{\mathrm{cyc}}^9$$，因此通过并行门缩短 cycle time 很重要。

## 如何实现并行两比特门？

考虑作用于离子 $$i,j$$ 的 SDF（spin-dependent force）：

$$
H(t)=\Omega(t)\sin(\mu t)\sum_{\ell\in\{i,j\}}\sum_m\eta_m b_{\ell m}X_\ell\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right).
$$

其中，$$\Omega(t)$$ 是由激光强度控制的驱动强度，$$\eta_m$$ 是第 $$m$$ 个振动模的 Lamb–Dicke 参数，$$b_{\ell m}$$ 是离子 $$\ell$$ 在该模中的参与系数，$$X_\ell$$ 是离子 $$\ell$$ 上的 Pauli 算符，$$a_m$$ 是第 $$m$$ 个模的声子湮灭算符，$$\omega_m$$ 是该模的频率。$$\mu$$ 是驱动频率，接近相关振动模的频率。忽略整体相位，时间演化可写为

$$
U(T)=\exp\!\left[\sum_{\ell\in\{i,j\}}\sum_m X_\ell\left(\alpha_{\ell m}a_m^\dagger-\alpha_{\ell m}^*a_m\right)+i\Theta_{ij}X_iX_j\right].
$$

先介绍两种两比特门方案。第一种是传统的 AM（amplitude modulation）：将 $$\Omega(t)$$ 划分为若干 time segment，每个 time segment 的强度保持不变，因此整个驱动波形可以用一个向量描述。我们需要优化这个向量，使纠缠角满足 $$\Theta_{ij}=\pi/4$$，同时使所有残余位移 $$\alpha_{\ell m}=0$$。

第二种是 AESE（adiabatic elimination of spin-motion entanglement）：令 $$\Omega(t)=\Omega_0\gamma(t)$$，其中 $$\gamma(t)$$ 是在开始和结束时平滑开启、关闭的 envelope，例如 $$\gamma(t)=\sin^2(\pi t/T)$$，$$T$$ 为门时间。当所有相关振动模满足 $$\vert \mu-\omega_m\vert T\gg1$$ 等绝热条件时，残余位移 $$\alpha_{\ell m}$$ 会受到抑制，具体要求见 [sutherland2024](https://arxiv.org/abs/2308.05865) 和 [hughes2025](https://arxiv.org/html/2510.17286v1#S2.SS2)。这样，我们只需调节 $$\Omega_0$$，使 $$\Theta_{ij}=\pi/4$$，无需逐段优化波形。这种方案对多种误差更鲁棒。在清华大学段组，我们认为 AESE 是实现离子阱两比特门的最佳方案。

![分段 AM 波形与 AESE 平滑 envelope 的对比](/images/blog/parallel-two-qubit-gates/waveform_comparison.png)

并行 AESE 门可以通过同时驱动两个 ion pair、为它们设置不同的频率 $$\mu_A$$ 和 $$\mu_B$$ 来实现。记两个 ion pair 分别为 $$A=\{i_A,j_A\}$$ 和 $$B=\{i_B,j_B\}$$，则

$$
\begin{aligned}
H(t)={}&\Omega_A(t)\sin(\mu_A t)\sum_{p\in A}\sum_m\eta_m b_{pm}X_p\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right)+\\
&\Omega_B(t)\sin(\mu_B t)\sum_{q\in B}\sum_m\eta_m b_{qm}X_q\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right).
\end{aligned}
$$

![以不同频率驱动两个 ion pair 的并行门示意图](/images/blog/parallel-two-qubit-gates/frequency_multiplexed_gate.png)

定义两组驱动的频率差为 $$\delta=\mu_A-\mu_B$$。其时间演化为

$$
\begin{aligned}
U(T)=\exp\!\Bigg[&\sum_{\ell\in A\cup B}\sum_m X_\ell\left(\alpha_{\ell m}a_m^\dagger-\alpha_{\ell m}^*a_m\right)+\\
&i\Theta_A X_{i_A}X_{j_A}+i\Theta_B X_{i_B}X_{j_B}+i\sum_{p\in A}\sum_{q\in B}\Theta_{pq}X_pX_q\Bigg].
\end{aligned}
$$

其中，残余位移 $$\alpha_{\ell m}$$ 仍可由 AESE 抑制，$$\Theta_A$$ 和 $$\Theta_B$$ 是两个 ion pair 各自需要的纠缠角，不同 ion pair 之间的 $$\Theta_{pq}$$ 则是 crosstalk。满足相应平滑性和 AESE 条件的 envelope 可使 crosstalk 相位随 $$1/(\vert \delta\vert T)^3$$ 衰减；对于 $$\sin^2$$ envelope，在我们关心的频率差范围内，可以进一步实现 $$1/(\vert \delta\vert T)^5$$ 的抑制。

![一般频率设置下的 crosstalk 纠缠角及斜率 −5 的参考线](/images/blog/parallel-two-qubit-gates/crosstalk_vs_delta.png)

这种方案的 crosstalk 抑制同样对噪声鲁棒。相比已有方案，比如 EASE [grzesiak2020](https://doi.org/10.1038/s41467-020-16790-9)，我们的方法对激光功率的要求更低，更多细节见 [tang2026](https://arxiv.org/abs/2609.04081)。

## 能达到什么指标？

我们在包含 512 个离子的二维晶体上进行了数值模拟，晶体参数参考 [guo2024](https://arxiv.org/abs/2311.17163)。通过合适的 ion–qubit mapping 和 schedule，我们发现，随着并行程度提高，BB248 的逻辑错误率显著下降，在现实噪声参数下可以达到 $$10^{-12}$$ 量级。

![BB248 每个 cycle 的逻辑错误率随激光功率变化](/images/blog/parallel-two-qubit-gates/bb248_logical_error.png)

其中，$$2\sum_k\Omega_k/(2\pi N)$$ 是激光功率的 proxy，也反映了并行程度。

![512 离子晶体中同时执行 124 个两比特门的 schedule 可视化](/images/blog/parallel-two-qubit-gates/bb248_parallel_schedule.png)

上图展示了最高并行度的 schedule：有连边的离子正在执行两比特门。在最高并行度下，单对门所需的 Rabi 频率 $$\Omega_k/(2\pi)$$ 仍为 $$5\,\mathrm{MHz}$$ 量级。

## Outlook

要在大规模二维离子晶格上实现 QEC，我们还需要解决两个问题：

1. crosstalk channel 的数量随并行程度呈平方增长，使纠错时需要考虑的 error columns 增多，进而影响 decoder 的表现。与此同时，我们的并行门能有效抑制 crosstalk，因此这些 channel 的发生概率很低。我们需要能够处理大量低概率 crosstalk channel 的 decoder。
2. 分析二维离子晶格中非线性耦合对两比特逻辑门及并行两比特门的影响，并在影响不可忽略时寻求解决方案。

我们正在推进这两个课题，预计将在未来一年内与大家分享成果。
