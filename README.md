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

POD-DeepONet improves upon Deep Operator Networks for partial differential equations by replacing the trunk network with proper orthogonal decomposition bases, but its single global basis proves insufficient for solutions with regime-dependent behavior.

We propose TEMPO, which discovers dynamical regimes via Expectation-Maximization in a low-dimensional POD coefficient space. Each mixture component constructs a locally optimal Neural-POD basis, while a learned gating network coordinates specialized branch networks via probabilistic mixture weights for smooth regime transitions.

Experiments on a range of multi-regime benchmarks demonstrate that automatic regime discovery with Neural-POD bases yields substantial accuracy gains over a single global basis.

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
