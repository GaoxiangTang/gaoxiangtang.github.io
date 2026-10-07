---
title: "Parallel two-qubit gates in ion traps and their application to QEC"
date: 2026-10-07 00:00:00 +0800
layout: post
lang: en
translation_key: parallel-two-qubit-gates-in-ion-traps
---

This blog introduces our paper [tang2026](https://arxiv.org/abs/2609.04081), which addresses three main questions:

1. Why does quantum error correction in ion traps require parallel two-qubit gates?
2. How can we implement parallel two-qubit gates with low crosstalk, robustness to noise, low power requirements, and simple waveform optimization?
3. What can we do with these parallel gates?

## Why quantum error correction in ion traps requires parallel two-qubit gates?

Consider the BB $$[248,10,18]$$ code [cain2026](https://arxiv.org/html/2603.28627v1#A2), which encodes 10 logical qubits in 248 physical qubits. Each cycle requires 1,488 two-qubit gates. Taking a two-qubit gate time of $$200\,\mu\mathrm{s}$$ [hughes2025](https://arxiv.org/html/2510.17286v1#S3.SS1), serial execution takes $$297.6\,\mathrm{ms}$$ per cycle. With $$T_2\approx3\,\mathrm{s}$$ [egan2021](https://arxiv.org/pdf/2009.11482#page=16), an exponential dephasing model predicts an average of about 12 Z flips from idle dephasing per cycle, exceeding the eight errors that a distance-18 code is guaranteed to correct.

Furthermore, for BB248 with distance 18, the logical error rate per cycle caused by idle dephasing scales as $$\epsilon_{\mathrm{cyc}}\sim t_{\mathrm{cyc}}^{\lfloor d/2\rfloor}=t_{\mathrm{cyc}}^9$$ at low error rates. Reducing cycle time through parallel gates is therefore important.

## How to implement parallel two-qubit gates?

Consider an SDF (spin-dependent force) acting on ions $$i,j$$:

$$
H(t)=\Omega(t)\sin(\mu t)\sum_{\ell\in\{i,j\}}\sum_m\eta_m b_{\ell m}X_\ell\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right).
$$

Here, $$\Omega(t)$$ is the drive amplitude, controlled by the laser intensity. $$\eta_m$$ is the Lamb–Dicke parameter of motional mode $$m$$, $$b_{\ell m}$$ is the participation coefficient of ion $$\ell$$ in that mode, $$X_\ell$$ is the Pauli operator on ion $$\ell$$, $$a_m$$ is the phonon annihilation operator for mode $$m$$, and $$\omega_m$$ is the mode frequency. The drive frequency $$\mu$$ is close to the frequencies of the relevant motional modes. Ignoring a global phase, the time evolution can be written as

$$
U(T)=\exp\!\left[\sum_{\ell\in\{i,j\}}\sum_m X_\ell\left(\alpha_{\ell m}a_m^\dagger-\alpha_{\ell m}^*a_m\right)+i\Theta_{ij}X_iX_j\right].
$$

We first introduce two approaches to two-qubit gates. The first is conventional AM (amplitude modulation): divide $$\Omega(t)$$ into time segments, each with a constant amplitude, so that the entire drive waveform can be represented by a vector. We optimize this vector to achieve the entangling angle $$\Theta_{ij}=\pi/4$$ while setting all residual displacements $$\alpha_{\ell m}=0$$.

The second is AESE (adiabatic elimination of spin-motion entanglement): set $$\Omega(t)=\Omega_0\gamma(t)$$, where $$\gamma(t)$$ is an envelope that turns on and off smoothly at the beginning and end of the gate. One example is $$\gamma(t)=\sin^2(\pi t/T)$$, where $$T$$ is the gate time. When all relevant motional modes satisfy adiabatic conditions such as $$\vert \mu-\omega_m\vert T\gg1$$, the residual displacements $$\alpha_{\ell m}$$ are suppressed. See [sutherland2024](https://arxiv.org/abs/2308.05865) and [hughes2025](https://arxiv.org/html/2510.17286v1#S2.SS2) for the specific requirements. We then only need to adjust $$\Omega_0$$ to achieve $$\Theta_{ij}=\pi/4$$, without optimizing each time segment of the waveform. This approach is more robust to multiple types of error. In Duan's group at Tsinghua University, we consider AESE the best approach to implementing two-qubit gates in ion traps.

![Comparison of a segmented AM waveform and a smooth AESE envelope](/images/blog/parallel-two-qubit-gates/waveform_comparison.png)

Parallel AESE gates can be implemented by simultaneously driving two ion pairs at different frequencies $$\mu_A$$ and $$\mu_B$$. Let the two ion pairs be $$A=\{i_A,j_A\}$$ and $$B=\{i_B,j_B\}$$. Then

$$
\begin{aligned}
H(t)={}&\Omega_A(t)\sin(\mu_A t)\sum_{p\in A}\sum_m\eta_m b_{pm}X_p\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right)+\\
&\Omega_B(t)\sin(\mu_B t)\sum_{q\in B}\sum_m\eta_m b_{qm}X_q\left(a_m e^{-i\omega_m t}+a_m^\dagger e^{i\omega_m t}\right).
\end{aligned}
$$

![Schematic of parallel gates driving two ion pairs at different frequencies](/images/blog/parallel-two-qubit-gates/frequency_multiplexed_gate.png)

Define the frequency difference between the two drives as $$\delta=\mu_A-\mu_B$$. The time evolution is

$$
\begin{aligned}
U(T)=\exp\!\Bigg[&\sum_{\ell\in A\cup B}\sum_m X_\ell\left(\alpha_{\ell m}a_m^\dagger-\alpha_{\ell m}^*a_m\right)+\\
&i\Theta_A X_{i_A}X_{j_A}+i\Theta_B X_{i_B}X_{j_B}+i\sum_{p\in A}\sum_{q\in B}\Theta_{pq}X_pX_q\Bigg].
\end{aligned}
$$

Here, the residual displacements $$\alpha_{\ell m}$$ can still be suppressed by AESE. $$\Theta_A$$ and $$\Theta_B$$ are the desired entangling angles for the two ion pairs, while $$\Theta_{pq}$$ between different ion pairs represents crosstalk. For envelopes satisfying the required smoothness and AESE conditions, the crosstalk phase decays as $$1/(\vert \delta\vert T)^3$$. For a $$\sin^2$$ envelope, the suppression improves to $$1/(\vert \delta\vert T)^5$$ over the frequency-difference range of interest.

![Crosstalk entangling angle at general frequency settings and a reference line with slope −5](/images/blog/parallel-two-qubit-gates/crosstalk_vs_delta.png)

The crosstalk suppression is also robust to noise. Compared with existing approaches such as EASE [grzesiak2020](https://doi.org/10.1038/s41467-020-16790-9), our method requires less laser power. See [tang2026](https://arxiv.org/abs/2609.04081) for more details.

## What performance can we achieve?

We performed numerical simulations of a two-dimensional crystal containing 512 ions, with crystal parameters based on [guo2024](https://arxiv.org/abs/2311.17163). With an appropriate ion–qubit mapping and schedule, we find that the logical error rate of BB248 decreases substantially as the degree of parallelism increases, reaching the $$10^{-12}$$ level under realistic noise parameters.

![BB248 logical error rate per cycle versus laser power](/images/blog/parallel-two-qubit-gates/bb248_logical_error.png)

Here, $$2\sum_k\Omega_k/(2\pi N)$$ is a proxy for laser power and also reflects the degree of parallelism.

![Schedule visualization of 124 simultaneous two-qubit gates in a 512-ion crystal](/images/blog/parallel-two-qubit-gates/bb248_parallel_schedule.png)

The figure above shows the schedule at the highest degree of parallelism. Connected ions are executing two-qubit gates. At this highest degree of parallelism, the Rabi frequency $$\Omega_k/(2\pi)$$ required for an individual gate remains on the order of $$5\,\mathrm{MHz}$$.

## Outlook

To implement QEC in large two-dimensional ion crystals, we still need to address two challenges:

1. The number of crosstalk channels grows quadratically with the degree of parallelism, increasing the number of error columns to account for in error correction and affecting decoder performance. At the same time, our parallel gates effectively suppress crosstalk, so these channels have very low occurrence probabilities. We need a decoder that can handle a large number of crosstalk channels with low probabilities.
2. Analyze the effects of nonlinear coupling in two-dimensional ion crystals on logical two-qubit gates and parallel two-qubit gates, and seek solutions if these effects are non-negligible.

We are working on both topics and expect to share the results within the coming year.
