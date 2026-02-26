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

Operator learning addresses the approximation of mappings between function spaces, which commonly arise in the solution of partial differential equations (PDEs) governing physical systems. Deep Operator Networks (DeepONet) learn such operators through branch and trunk neural networks. POD-DeepONet improves upon this by replacing the trunk network with Proper Orthogonal Decomposition (POD) bases derived from solution snapshots, achieving superior accuracy and training efficiency.
However, existing POD-DeepONet methods employ a single global basis across all parameters, which proves insufficient when solutions exhibit regime-dependent behavior such as laminar-turbulent transitions or diffusion-reaction regime changes. We propose TEMPO (EM-based Mixture-of-PODs), an adaptive framework combining Expectation-Maximization clustering with multiple regime-specific POD bases. Gaussian Mixture Models automatically discover distinct dynamical regimes by partitioning solution snapshots in a low-dimensional POD coefficient space. For each discovered regime, we construct locally optimal POD bases that efficiently capture regime-specific solution features.
The architecture employs multiple regime-conditioned branch networks, each paired with its corresponding POD basis, coordinated through a learned gating network that produces probabilistic mixture weights for smooth regime transitions. This approach enables automatic regime discovery without manual partitioning, achieves superior local approximation through specialized bases, and provides parametric adaptivity where regime selection depends on input conditions. We validate the framework on benchmark PDEs exhibiting multi-regime phenomena, demonstrating significant improvements over standard POD-DeepONet for complex operators.

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