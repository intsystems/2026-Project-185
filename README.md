# Operator learning in PINN
-----
### TEMPO: EM-Based Mixture-of-PODs for Operator Learning

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

Deep Operator Networks (DeepONet) learn mappings between function spaces through branch and trunk neural networks. POD-DeepONet improves upon this by replacing the trunk network with Proper Orthogonal Decomposition bases derived from solution snapshots, achieving superior accuracy and training efficiency. However, existing POD-DeepONet methods employ a single global basis across all parameters, which proves insufficient when solutions exhibit regime-dependent behavior.
We propose an adaptive POD-DeepONet framework combining Expectation-Maximization clustering with multiple regime-specific bases. Gaussian Mixture Models automatically identify distinct dynamical regimes by partitioning solution snapshots. For each regime, we construct locally optimal POD bases that capture regime-specific features, where modern Neural-POD approaches can be employed for basis construction. The architecture features multiple cluster-conditioned trunks with probabilistic mixture weights for smooth regime transitions, while branch networks predict regime-specific coefficients.
This framework enables automatic regime discovery, achieves superior local approximation through specialized bases, reduces dimensionality requirements per regime, and provides natural uncertainty quantification. The probabilistic formulation ensures continuous predictions across regime boundaries without manual domain partitioning.
We validate the approach on PDEs exhibiting multi-scale and regime-dependent phenomena, demonstrating significant improvements over standard POD-DeepONet for complex operators.

## Keywords
Operator learning, DeepONet, Proper Orthogonal Decomposition, Expectation-Maximization, Gaussian mixture models, Neural-POD

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