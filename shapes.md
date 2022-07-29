# Purpose

This document lists the shape of weights and biases, and other important tensors.

### Weights

If a layer receives $m$ values as inputs and outputs $n$ values as outputs, then its weight matrix shape is $m \times n$.

$$\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

where $a_{ij}$ is the contribution from the $j$-th component of the input to the $i$-th component of the output.

If the layer is flattened by reshaping its size to $-1$, then the matrix becomes $(a_{11}, \cdots, a_{1n}, a_{21}, \cdots)$ etc.

### Bias

If a layer outputs $n$ values, then its bias vector has $n$ components. It represents the bias added to the $i$-th component of the output.