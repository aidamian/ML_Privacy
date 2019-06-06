# Differential Privacy examples repository

Requirements:
- PyTorch 1.
- PySyft

##Notes:
`test_03` actually simultates a full process of applying differential privacy techniques to Deep Learning. The used dataset is `MNIST` and we assume we have various numbers of localized databases where we can train our model and we also vary the privacy leakeage threshold. The final results are below:

```
    DPModel     Model  Workers   Eps
0  0.135650  0.988433        2  0.01
3  0.296983  0.989917        7  0.01
6  0.390350  0.991200       15  0.01
1  0.391533  0.988250        2  0.10
4  0.928200  0.990733        7  0.10
7  0.958133  0.991417       15  0.10
8  0.967183  0.990900       15  0.80
2  0.971417  0.988233        2  0.80
5  0.977867  0.988900        7  0.80


    DPModel     Model  Workers   Eps
   0.630767  0.990167       30  0.01
   0.731600  0.990883       80  0.01
   0.739117  0.989600       40  0.01


```

