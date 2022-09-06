# JUMBOT
Official Python3 implementation of our ICML 2021 paper "Unbalanced minibatch Optimal Transport; applications to Domain Adaptation".

### Python Optimal Transport (POT) 0.8.x
!!! The following code is based on POT 0.7. Check the version you are using before using this code. If you want to use POT 0.8.x, you have to convert input probability weights and ground cost to Float64 otherwise it will not work.

```
a, b = ot.unif(g_xs.size()[0]).astype('float64'), ot.unif(g_xt.size()[0]).astype('float64')
pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.double().detach().cpu().numpy(),
                                             0.01, reg_m=reg_m) ## POT 0.8.2
```


### Abstract
Optimal transport distances have found many applications in machine learning for their capacity to compare non-parametric probability distributions. Yet their algorithmic complexity generally prevents their direct use on large scale datasets. Among the possible strategies to alleviate this issue, practitioners can rely on computing estimates of these distances over subsets of data, {\em i.e.} minibatches. While computationally appealing, we highlight in this paper some limits of this strategy, arguing it can lead to undesirable smoothing effects. As an alternative, we suggest that the same minibatch strategy coupled with unbalanced optimal transport can yield more robust behavior. We discuss the associated theoretical properties, such as unbiased estimators, existence of gradients and concentration bounds. Our experimental study shows that in challenging problems associated to domain adaptation, the use of unbalanced optimal transport leads to significantly better results, competing with or surpassing recent baselines.

### How to cite
This paper has been accepted to [ICML 2021](https://icml.cc/Conferences/2021). If you use this toolbox in your research or unbalanced minibatch OT and find them useful, please cite unbalanced minibatch OT using the following bibtex reference:

```
@InProceedings{fatras2021jumbot,
author    = {Fatras, Kilian and S\'ejourn\'e, Thibault and Courty, Nicolas and Flamary, R\'emi},
title     = {Unbalanced minibatch Optimal Transport; applications to Domain Adaptation},
booktitle = {Proceedings of the 38th International Conference on Machine Learning},
year      = {2021}
}
```

If you use JUMBOT in your research or minibatch Unbalanced OT and find them useful, please also cite "Minibatch optimal transport distances; analysis and applications" and "Learning with minibatch Wasserstein: asymptotic and gradient properties" as JUMBOT is based on them. You can use the following bibtex references:

```
@misc{fatras2021minibatch,
      title={Minibatch optimal transport distances; analysis and applications}, 
      author={Kilian Fatras and Younes Zine and Szymon Majewski and Rémi Flamary and Rémi Gribonval and Nicolas Courty},
      year={2021},
      eprint={2101.01792},
      archivePrefix={arXiv},
}
```

```
@InProceedings{fatras2019learnwass,
author    = {Fatras, Kilian and Zine, Younes and Flamary, Rémi and Gribonval, Rémi and Courty, Nicolas},
title     = {Learning with minibatch Wasserstein: asymptotic and gradient properties},
booktitle = {AISTATS},
year      = {2020}
}
```


### Prerequisites

* Python3 (3.7.3)
* PyTorch (1.6.0)
* POT (0.7.0)
* Numpy (1.16.4)
* Scipy (1.2.0)
* argparse (1.1)
* os
* CUDA


### What is included ?

* (I am currently writing my PhD thesis and the calendar might change.)
* JUMBOT on digits (SVHN to MNIST)
* JUMBOT on Office-Home and VisDA
* JUMBOT on Partial Office-Home


### Paper authors

* [Kilian Fatras](https://kilianfatras.github.io/)
* [Thibault Séjourné](https://thibsej.github.io/)
* [Nicolas Courty](https://github.com/ncourty)
* [Rémi Flamary](http://remi.flamary.com/)
