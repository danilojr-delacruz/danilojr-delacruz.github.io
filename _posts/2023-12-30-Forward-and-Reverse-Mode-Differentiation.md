---
title: Forward and Reverse Mode Differentiation
date: 2023-12-30
categories: [Machine Learning, Implementation]
tags: [ml]     # TAG names should always be lowercase
publish: true
math: true
---

<!-- TODO: Can we make this a call out? -->
> Methods to efficiently compute gradients are important as they underpin the training of Machine Learning Models. In particular, we compare Forward and Reverse Mode Differentiation for gradient estimation of function compositions. The ubiquitous Backpropagation Algorithm is Reverse Mode Differentiation applied to Neural Networks and we will show why we typically ignore its forward counterpart.

## Background
Given functions $$f_{0}, f_{1}, \dots, f_{m}$$ with known Jacobians, where $$f_{0}(x) = x$$. We want to compute the Jacobian of $$f$$.

$$
f_{i}:\mathbb{R}^{d_{i-1}} \to \mathbb{R}^{d_{i}},\quad
f(x) = f_{m} \circ \cdots \circ f_{1} (x)
$$

By the Chain Rule,

$$
\underset{ d_{m} \times d_{0} }{ \frac{df}{dx}
\vphantom{\frac{df_{m}}{df_{m-1}}}
}
=
\underset{ d_{m} \times d_{m-1}  }{ \frac{df_{m}}{df_{m-1}}
\vphantom{\frac{df_{m}}{df_{m-1}}}
}
\cdots
\underset{ d_{1} \times d_{0} }{ \frac{df_{1}}{df_{0}}
\vphantom{\frac{df_{m}}{df_{m-1}}}
}
= \prod_{i=1}^{m+1} \frac{df_{i}}{df_{i-1}}.
$$

Now it remains to establish a scheme to compute $$\frac{df}{dx}$$ efficiently.

> Recall that Jacobian of $$f : \mathbb{R}^{d_{0}} \to \mathbb{R}^{d_{m}}$$ is defined as
>
> $$
\left(\frac{df}{dx}\right)_{ij} = \frac{\partial f_{i}}{\partial x_{j}},\quad
\frac{df}{dx} =
\underset{ d_{m} \times d_{0} }{ \begin{bmatrix}
\vdots & \vdots & \vdots \\
\frac{\partial f}{\partial x_{1}} & \cdots & \frac{\partial f}{\partial x_{d_{0}}} \\
\vdots & \vdots & \vdots
\end{bmatrix}. }
> $$
>
> See Notation at end of document for clarification on what is meant by $$\frac{df_{i}}{df_{i-1}}$$.
<!-- Can't link internally? -->

### Forward-Mode Differentiation
This method studies how the intermediate layers $$f_{i}$$ vary with the initial layer $$x$$. We apply a forward recursion until we obtain the Jacobian of the final layer $$f$$. In particular, for $$i = 0, \dots, m-1$$

$$
\underset{ d_{i+1} \times d_{0} }{ \frac{df_{i+1}}{dx}
\vphantom{\frac{df_{i+1}}{df_{i}}}
}
=
\underset{ d_{i+1} \times d_{i} }{ \frac{df_{i+1}}{df_{i}}
\vphantom{\frac{df_{i+1}}{df_{i}}}
}
\,
\underset{ d_{i} \times d_{0} }{ \frac{df_{i}}{dx}
\vphantom{\frac{df_{i+1}}{df_{i}}}
}
$$

The algorithm comprises of one pass which computes $$f(x)$$ and $$\frac{df}{dx}$$
- Base Case $$f_{0} = x$$ and $$\frac{df_{0}}{dx} = I$$ (Identity matrix)
- Iteration $$i$$:
    - Recall $$x_{i}$$ and $$\frac{df_{i}}{dx}$$ from the previous iteration
    - Compute $$x_{i+1} = f_{i+1}(x_{i})$$ and "transition jacobians" $$\frac{df_{i+1}}{df_{i}}(x_{i})$$
    - Compute $$\frac{df_{i+1}}{dx} = \frac{df_{i+1}}{df_{i}} \frac{df_{i}}{dx}$$
    - Set $$i = i + 1$$
- At iteration $$m$$, return $$\frac{df_{m}}{dx}$$

### Reverse-Mode Differentiation
This method studies how the output $$f$$ varies with the intermediate layers $$f_{i}$$. We apply a backward recursion until we obtain the Jacobian with respect to the initial layer $$x$$. In particular for $$i = 0, \dots, m-1$$, we use the formula

$$
\underset{ d_{m} \times d_{m-i-1} }{ \frac{df_{m}}{df_{m-i - 1}}
\vphantom{\frac{df_{m}}{df_{m-i}}}
}
=
\underset{ d_{m} \times d_{m-i} }{ \frac{df_{m}}{df_{m-i}}
\vphantom{\frac{df_{m}}{df_{m-i}}}
}
\;
\underset{ d_{m-i} \times d_{m-i-1} }{ \frac{df_{m-i}}{df_{m-i-1}}
\vphantom{\frac{df_{m}}{df_{m-i}}}
}
$$

The algorithm comprises of two passes.
- The forward pass computes $$f_{i}$$ and the "transition jacobians" $$\frac{df_{i}}{df_{i-1}}$$
    - Same as forward mode but we do not compute $$\frac{df_{i+1}}{dx}$$
- The backward pass computes $$\frac{df_{m}}{dx}$$.
    - Base case $$\frac{df_{m}}{df_{m}} = I$$
    - At iteration $$i$$:
        - Compute $$\frac{df_{m}}{df_{m-i-1}} = \frac{df_{m}}{df_{m-i}} \frac{df_{m-i}}{df_{m-i-1}}$$
        - Set $$i = i + 1$$
    - At iteration $$m$$ return $$\frac{df_{m}}{df_{0}} = \frac{df_{m}}{dx}$$

## Comparison
The direction of recursion in forward and backward mode yields the method's name but it also results in the cost being dependent on the input and output dimensions. In general, a recursion comprises of Matrix-Matrix Multiplication which is $$O(d^{3})$$ where $$d = \max d_{i}$$. However
- If $$d_{0} = 1$$ and $$x$$ is a scalar, then forward recursion comprises of a Matrix-Vector Multiplication, $$O(n^{2})$$
- Likewise if $$d_{m} = 1$$ and $$f$$ is a scalar, then backward recursion comprises of Vector-Matrix Multiplication, also $$O(n^{2})$$.

In the context of Neural Networks, $$f$$ typically corresponds to a 1-dimensional loss function (e.g. Cross-Entropy for classification and Mean-Square Error in regression) and this is the case we will consider.

### Computational-Cost
In terms of function evaluations and computation of transition jacobians, Forward and Backward mode are the same. We only need to consider the cost of the recursion.

For Forward mode this is $$O(md^{3})$$ whereas only having to perform Vector-Matrix Multiplication reduces the complexity of Backward Mode to $$O(md^{2})$$.
### Numerical Stability
This discrepancy in computational cost is closely related to the common trick used in Numerical Linear Algebra where $$ABx$$ is evaluated as $$A(Bx)$$ as opposed to $$(AB)x$$.

The other reason this is beneficial, is because Matrix-Matrix multiplication is not numerically (backward) stable whereas Matrix-Vector Multiplication is. See [Section 7.6 Oxford Maths C6.1 Numerical Linear Algebra Notes (2023-2024)](https://courses.maths.ox.ac.uk/course/view.php?id=5024#section-1). This means that a naive implementation of Forward Mode would be unstable.

The stability of matrix multiplication can be improved for a slight increase in computational complexity, see [Fast linear algebra is stable - James Demel, Ioana Dumitriu, Olga Holtz (2007)](https://arxiv.org/abs/math/0612264). Despite, the paper claiming these algorithms are parallelisable, it is unlikely to have an implementation with minimised constant factors due to its age. In particular this means, the increase in computing time would be exacerbated by a difference in software.

These two reasons demonstrate why Backward Mode is appealing. However the next section, demonstrates its major caveat.
### Memory-Cost
In Forward Mode, we only need to store the previous transition jacobians $$\frac{df_{i+1}}{df_{u}}$$ which yields a memory cost of $$O(d^{2})$$.

On the other hand, Backward Mode needs to store all the transition jacobians during the forward pass as it can only use them during the backward pass. This leads to a memory cost of $$O(md^{3})$$.

As the computational powers of computers are plateauing and applications often call for neural networks with millions of parameters, memory can be a bottleneck. There has been some attempts to alleviate this.
- [Memory-Efficient Backpropagation Through Time - Deepmind (2016)](https://arxiv.org/abs/1606.03401) proposed a method which decreased the memory usage by 95% for backpropagation in a recurrent neural network. For recurrent neural networks, because the depth $$m$$ is large, the memory cost is acutely felt.
    - (Unsure: I will aim to discuss this paper in another post. But I think the idea is to recompute the earlier gradients from scratch in the backward pass as opposed to holding them for the entire run. This increases computational cost.)
- [The Symplectic Adjoint Method: Memory-Efficient Backpropagation of Neural-Network-Based Differential Equations - Takashi Matsubara, Yuto Miyatake, Takaharu Yaguchi (2023)](https://ieeexplore.ieee.org/document/10045756)
    - Here $$du = f_{\theta}(u) dt$$ where $$f_{\theta}$$ is a neural network.
    - If we perform numerical integration, then we need to evaluate $$f_{\theta}$$ say $$n$$ times. Then we will need to hold $$O(nmd^{3})$$ in memory.
    - The adjoint method only requires one evaluation but is numerically unstable.
    - Their new method retains the computational complexity of the adjoint method whilst retaining numerical stability.

These two papers indicate that academia is starting to view memory cost as an important consideration when designing algorithms.

## Conclusion
Forward Mode computes gradients while it evaluates the function $$f$$ whereas Backward Mode computes gradients in reverse after evaluation.

Since $$f$$ is a typically a scalar, Forward Mode takes longer to run whereas Backward Mode requires more memory.

In the past, the speed of training neural networks has been the main concern. The application of the Backpropagation algorithm and advances in GPU-acceleration were necessary to demonstrate the feasibility of Neural networks, see [AlexNet](https://en.wikipedia.org/wiki/AlexNet). However, memory limitations are now emerging as a concern and practitioners will need to find a way to balance the trade-off between Computational Time and Memory Space.

|Mode|Computation|Memory|Stability|
|-|-|-|-|
|Forward|$$O(md^{3})$$|$$O(d^{3})$$|Unstable|
|Backward|$$O(md^{2})$$|$$O(md^{3})$$|Stable|

<!-- Better way to do referencing in markdown? -->
## References
- Appendix A.1 in [On Neural Differential Equations - Patrick Kidger (2022)](https://arxiv.org/pdf/2202.02435.pdf) for an introduction on Forward and Reverse mode differentiation.
- [Section 7.6 Oxford Maths C6.1 Numerical Linear Algebra Notes (2023-2024)](https://courses.maths.ox.ac.uk/course/view.php?id=5024#section-1)
- [Fast linear algebra is stable - James Demel, Ioana Dumitriu, Olga Holtz (2007)](https://arxiv.org/abs/math/0612264)
- [Memory-Efficient Backpropagation Through Time - Deepmind (2016)](https://arxiv.org/abs/1606.03401)
- [The Symplectic Adjoint Method: Memory-Efficient Backpropagation of Neural-Network-Based Differential Equations - Takashi Matsubara, Yuto Miyatake, Takaharu Yaguchi (2023)](https://ieeexplore.ieee.org/document/10045756)
- [AlexNet](https://en.wikipedia.org/wiki/AlexNet)

## Notation
As an example, let us take $$m=3$$ where $$f = f_{3} \circ f_{2} \circ f_{1} \circ x$$ and let us use $$x_{i} \in \mathbb{R}^{d_{i-1}}$$ as a dummy variable to represent the input of $$f_{i}$$.
Formally, Chain Rule is stated as

$$
\left. \frac{df}{dx} \right \rvert_{x}
=
\left. \frac{df_{3}}{dx_{3}} \right \rvert_{x_{3} = f_{2} \circ f_{1} \circ x}
\left. \frac{df_{2}}{dx_{2}} \right \rvert_{x_{2} = f_{1} \circ x}
\left. \frac{df_{1}}{dx_{1}} \right \rvert_{x_1 =  x}.
$$

This is quite burdensome to write and by abuse of notation,

$$
\frac{df}{dx}
=
\frac{df_{3}}{df_{2}}
\frac{df_{2}}{df_{1}}
\frac{df_{1}}{dx}.
$$

Nevertheless it is somewhat fitting as our evaluation point is a function of $$f_{i-1}$$ hence $$f_{i-1}$$ has the correct dimension for $$x_{i}$$.