# TEMPO: Regime-Aware Operator Learning with Locally Adapted POD Bases
-----

[![License](https://badgen.net/github/license/intsystems/2026-Project-185?color=green)](https://github.com/intsystems/2026-Project-185/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/2026-Project-185)](https://github.com/intsystems/2026-Project-185/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/intsystems/2026-Project-185.svg?color=0088ff)](https://github.com/intsystems/2026-Project-185/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/intsystems/2026-Project-185.svg?color=7f29d6)](https://github.com/intsystems/2026-Project-185/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Rodion Akinzhala</td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Alexander Terentyev </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td>  Vadim Strijov, DSc</td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Operator learning methods typically approximate the solution map of a physical system with a single global model — an assumption that fails when solutions change character fundamentally across the parameter space.

We propose TEMPO: a framework that identifies latent dynamical regimes in snapshot data via Expectation-Maximization, builds a dedicated Neural-POD basis for each, and learns to route new inputs to the right regime at inference time. On multi-regime benchmarks, TEMPO achieves substantially lower errors than POD-DeepONet and vanilla DeepONet, with interpretable structure that reflects the underlying physics.

## Keywords
Operator learning, DeepONet, Proper Orthogonal Decomposition, Expectation-Maximization, Gaussian mixture models, Regime discovery

