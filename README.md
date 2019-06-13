# Differential Privacy examples repository

Requirements:
- PyTorch 1.x
- PySyft

##Notes:
- `test_01` it's a simple naive implementation of local differential privacy
- `test_03` similar to 1st test the second simulates the global DP
- `test_03` actually simultates a full process of applying differential privacy techniques to Deep Learning. The used dataset is `MNIST` and we assume we have various numbers of localized databases where we can train our model and we also vary the privacy leakeage threshold. The final results are below.
Initial simple split train in workers (data overlapping between remotes and local  but no overlapping between remotes) then train on full with labels:

```

            DPModel     Model  Workers   Eps
           0.135650  0.988433        2  0.01
           0.296983  0.989917        7  0.01
           0.390350  0.991200       15  0.01
           0.391533  0.988250        2  0.10
           0.928200  0.990733        7  0.10
           0.958133  0.991417       15  0.10
           0.967183  0.990900       15  0.80
           0.971417  0.988233        2  0.80
           0.977867  0.988900        7  0.80
        
        
            DPModel     Model  Workers   Eps
           0.630767  0.990167       30  0.01
           0.731600  0.990883       80  0.01
           0.739117  0.989600       40  0.01
```

The more advanced tests was based on retaining 40% of data for the local model and distributing 60% of data equaly to all remote sites.  No data overlapping between any site.        

```
            DPModel     Model  Workers  Eps (NLL)
           0.835567  0.990533        7  0.1
           0.928650  0.987933       15  0.1
           
            DPModel     Model  Workers  Eps (CCE)     
           0.782233  0.995150        7  0.1
           0.925133  0.994183       15  0.1


```

