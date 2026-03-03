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

Operator learning methods for partial differential equations based on Deep Operator Networks have been substantially improved by POD-DeepONet, which replaces the learned trunk network with precomputed proper orthogonal decomposition bases derived from solution snapshots. However, POD-DeepONet employs a single global basis shared across the entire solution space, which proves insufficient when solutions exhibit regime-dependent behavior such as laminar-turbulent transitions or diffusion-reaction regime changes. 

We propose TEMPO — an extension of POD-DeepONet that automatically discovers dynamical regimes via Expectation-Maximization in a low-dimensional POD coefficient space. Each mixture component yields a locally optimal basis tailored to its regime, while a learned gating network maps input parameters to probabilistic mixture weights at inference, coordinating an ensemble of specialized branch networks for smooth transitions across regimes.

We validate TEMPO on benchmark partial differential equations exhibiting multi-regime phenomena, demonstrating that regime-aware basis adaptation yields systematic accuracy improvements over POD-DeepONet on complex multi-regime operators.

## Keywords
Operator learning, DeepONet, Proper Orthogonal Decomposition, Expectation-Maximization, Gaussian mixture models, Regime discovery

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
