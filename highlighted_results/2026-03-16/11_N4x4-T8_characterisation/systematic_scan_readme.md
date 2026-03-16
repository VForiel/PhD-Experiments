
systematic_scan.npz should be read using the `np.load` function :

```python
data = np.load("systematic_scan.npz")
```

The first dimension of the arrays is the shifter index (from 0 to 3), then the output and finally the phase injected
