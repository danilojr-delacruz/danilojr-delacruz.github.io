---
title: Forward and Reverse Mode Differentiation
date: 2023-12-30
categories: [Machine Learning, Implementation]
tags: [ml]     # TAG names should always be lowercase
publish: true
math: true
---

This is still a draft.

We have a function $$f = f_{m} \circ \cdots \circ f_{1} \circ x$$ obtained via compositions of functions and we want to compute the Jacobian.
- Forward-Mode Differentiation computes the Jacobian by studying how the intermediate layers vary with the input until we obtain the final layer. (Recursion moves forward)
- Reverse-Mode Differentiation computes the Jacobian by studying how the output varies with the intermediate layers until we obtain the initial layer. (Recursion goes backwards)
In the case that $$f$$ is a scalar, Reverse-Mode Differentiation is computationally cheaper and numerically stable. However it incurs a greater memory cost as it has to keep track of all the intermediate Jacobians until the reverse pass.



Suppose that we have functions $$f_{1}, \dots, f_{m}$$, setting $$f_{0}(x) = x$$. We want to compute the Jacobian of $$f$$.

$$
f_{i}:\mathbb{R}^{d_{i-1}} \to \mathbb{R}^{d_{i}},\quad
f(x) = f_{m} \circ \cdots \circ f_{1} (x)
$$

By the Chain Rule we have that (TODO: align these undersets some other time)

$$
\underset{ d_{m} \times d_{0} }{ \frac{df}{dx} }
=
\underset{ d_{m} \times d_{m-1}  }{ \frac{df_{m}}{df_{m-1}} }
\cdots
\underset{ d_{1} \times d_{0} }{ \frac{df_{1}}{df_{0}} }
= \prod_{i=1}^{m+1} \frac{df_{i}}{df_{i-1}}
$$

There are different ways to compute these

## Forward-Mode Differentiation
Here we study how the intermediate layers vary with the initial layer (input). Then we propagate to the final layer which yields the desired answer.
In particular we apply the formula for $$q = 0, \dots, m-1$$

$$
\frac{df_{q+1}}{dx} = \frac{df_{q+1}}{df_{q}} \frac{df_{q}}{dx}
$$

Algorithm:
- Compute $$f_{1}(x)$$ and the jacobian $$\frac{df_{1}}{dx}$
- Then compute $$f_{2}(x')$$ with $$x' = f_{1}(x)$$
    - Compute Jacobian $$\frac{df_{2}}{dx'} = \frac{df_{2}}{df_{1}}$$
    - Multiply with previous value to get $$\frac{df_{2}}{dx}$$
- Repeat

## Reverse-Mode Differentiation
We study how the final layer (output) vary with the intermediate layers. Then we propagate backwards to the initial layer which yields the desired answer.
In particular for $$q = 1, \dots, m$$, we use the formula

$$
\frac{df_{m}}{df_{q-1}} = \frac{df_{m}}{df_{q}} \frac{df_{q}}{df_{q-1}}
$$

Algorithm:
- Forward pass: Compute $$f_{i}$$ and track the jacobians $$\frac{df_{q}}{df_{q-1}}$
- At the end compute $$\frac{df_{m}}{df_{m-1}}$
- Backward Pass: Propagate all the way back to obtain $$\frac{df_{m}}{dx}$

## Comparison
### Direction
- Forward: q increases
- Backward: q decreases

### Memory Cost
For backwards we need to store all the Jacobians until we get to the end.
This means that we need to store $$O\left( \sum d_{i} \times d_{i-1} \right) = O\left(m \max d_{i} \times d_{i-1} \right)$$ entries.
This can get really bad when we have a large number of functions.

Whereas forwards only requires us to store the current and transition jacobian.
Leading to a storage cost of $$O(\max d_{i} \times d_{i-1})$

### Computational Cost
In the special case that $$d_{m} = 1$$, i.e. $$f$ is a scalar function then
- $$\frac{df_{m}}{df_{q}}$$ is a vector as it has dimension $$1 \times d_{q}$$. Hence Backwards has an edge as only need to perform vector-matrix multiplication leading to a cost $$O(m \max d_{i}^{2})$
- Whereas Forwards is always matrix-matrix multiplication. Leading to a cost of $$O(m \max d_{i}^{3})$

### Numerical Stability
Still in the case of a scalar output function. Backwards is numerically stable, whereas forwards is not.
Recall that [[Matrix-Matrix Multiplication is Not Numerically Stable]] whereas [[Matrix-Vector Multiplication is Numerically Stable]]

### Remark on scalar output function
This is actually pretty common. Often we are trying to minimise a loss function which is a scalar.

## Remark on notation
We are being slightly sloppy with notation.
For example let us take $$m=3$$ where $$f = f_{3} \circ f_{2} \circ f_{1} \circ x$
The Chain Rule formula is actually

$$
\frac{df}{dx}
=
\left. \frac{df_{3}}{dx_{3}} \right \rvert_{x_{3} = f_{2} \circ f_{1} \circ x}
\left. \frac{df_{2}}{dx_{2}} \right \rvert_{x_{2} = f_{1} \circ x}
\left. \frac{df_{1}}{dx_{1}} \right \rvert_{x_1 =  x}
$$

But this is quite burdensome to write and we just do

$$
\frac{df}{dx}
=
\frac{df_{3}}{df_{2}}
\frac{df_{2}}{df_{1}}
\frac{df_{1}}{dx}
$$

# References