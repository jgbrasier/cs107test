### Multivariate, vector-valued function

This is not supported. Be wary when calling the function draw_graph_expensive on an array of node, as the resulting graph corresponds to the result of multiplying the Jacobian of the function under consideration by a vector of ones. 

Consider the function:

<img src="https://latex.codecogs.com/svg.image?f:&space;\mathbb{R}^3&space;\rightarrow&space;\mathbb{R}^3&space;" title="f: \mathbb{R}^3 \rightarrow \mathbb{R}^3 " />

<img src="https://latex.codecogs.com/svg.image?f:&space;\begin{pmatrix}&space;x&space;\\&space;y&space;\\&space;z&space;\end{pmatrix}&space;\mapsto&space;\begin{pmatrix}&space;2x&plus;y&space;\\&space;xy&plus;x&space;\\&space;xz&plus;x&space;\end{pmatrix}&space;" title="f: \begin{pmatrix} x \\ y \\ z \end{pmatrix} \mapsto \begin{pmatrix} 2x+y \\ xy+x \\ xz+x \end{pmatrix} " />

Then:

<img src="https://latex.codecogs.com/svg.image?J_f=\begin{pmatrix}&space;2&space;&&space;1&space;&&space;0&space;\\&space;y&plus;1&space;&&space;x&space;&&space;0&space;\\&space;z&plus;1&space;&&space;0&space;&&space;x&space;&space;\end{pmatrix}&space;" title="J_f=\begin{pmatrix} 2 & 1 & 0 \\ y+1 & x & 0 \\ z+1 & 0 & x \end{pmatrix} " />

<img src="https://latex.codecogs.com/svg.image?J_f|_{(1,2,0)}=\begin{pmatrix}&space;2&space;&&space;1&space;&&space;0&space;\\&space;3&space;&&space;1&space;&&space;0&space;\\&space;1&space;&&space;0&space;&&space;1&space;&space;\end{pmatrix}&space;" title="J_f|_{(1,2,0)}=\begin{pmatrix} 2 & 1 & 0 \\ 3 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix} " />

Let us vizualise this.

This is test_viz_a8 in test_viz_expensive.py, just set draw_=True when calling the function if you want to reproduce it.

```python
x=variable(1)
y=variable(2)
z=variable(0)
f1=x+x+y
f2=x*y+x 
f3=x*z+x
draw_graph_expensive([f1, f2, f3])
```

![](figs/Figure_super_complex.png)

Calling draw_reverse_graph for the function f returns a graph that corresponds to:

<img src="https://latex.codecogs.com/svg.image?J_f|_{(1,2,0)}=\begin{pmatrix}1&space;&&space;1&space;&1&space;&space;\end{pmatrix}\begin{pmatrix}&space;2&space;&&space;1&space;&&space;0&space;\\&space;3&space;&&space;1&space;&&space;0&space;\\&space;1&space;&&space;0&space;&&space;1&space;&space;\end{pmatrix}=\begin{pmatrix}&space;6&space;&&space;2&space;&&space;1&space;\end{pmatrix}&space;&space;" title="J_f|_{(1,2,0)}=\begin{pmatrix}1 & 1 &1 \end{pmatrix}\begin{pmatrix} 2 & 1 & 0 \\ 3 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}=\begin{pmatrix} 6 & 2 & 1 \end{pmatrix} " />

This is test_reverse_a3 in test_vix_reverse.py (just set draw_=True when calling the function).

```python
x=variable(1)
y=variable(2)
z=variable(0)
f1=x+x+y
f2=x*y+x 
f3=x*z+x
draw_reverse_graph([f1, f2, f3])
```

![](figs/Figure_rev_very_complex.png)





