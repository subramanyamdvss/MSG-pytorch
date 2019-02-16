# MSG-pytorch
  This repository contains pytorch implementation of all stochastic variants of PCA. Run demo.sh for a demo.
  The following table depicts results for a sample dataset. 

| Method | Train obj (higher is good) | Val obj |
| --- | --- | --- |
| L1-L2-MSG | 229447.9688 | 226028.4219 |
| Incremental | 220103.4219 | 216784.5156 |
| Stochastic Power | 219609.2500 | 216295.8750 |
| Original | 219832.8281 | 216516.5000 |

##### References
[1] Mianjy, Poorya, and Raman Arora. "Stochastic PCA with $\ell_2 $ and $\ell_1 $ Regularization." International Conference on Machine Learning. 2018.
[2] Raman Arora, Andrew Cotter, Karen Livescu, and Nathan Srebro. Stochastic optimization for
pca and pls. In Communication, Control, and Computing (Allerton), 2012 50th Annual Allerton
Conference on, pages 861–868. IEEE, 2012.
[3] Raman Arora, Andy Cotter, and Nati Srebro. Stochastic optimization of pca with capped msg. In
Advances in Neural Information Processing Systems, pages 1815–1823, 2013.
[4] Manfred K Warmuth and Dima Kuzmin. Randomized online pca algorithms with regret bounds
that are logarithmic in the dimension. Journal of Machine Learning Research, 9(Oct):2287–2320,
2008.
[5] Amir Beck and Marc Teboulle. Mirror descent and nonlinear projected subgradient methods for
convex optimization. Operations Research Letters, 31(3):167–175, 2003.
