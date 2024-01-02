---
title: Automatic Differentiation
date: 2024-01-02
categories: [Machine Learning, Implementation]
tags: [ml]     # TAG names should always be lowercase
publish: false
math: true
---

TODO:
- Understand what pytorch does for autograd. Does it even do checkpoints.? What's the reentrant thing and hooks?
- Finish understanding the recursive version. And then the thing about realistic assumptions due to non-uniformity of memory cost.

{% include admonition.html type="abstract" title="Summary" body="
Abstract here.
"
%}

Automatic Differentiation (AD) is a procedure for computing derivatives of a function $f$ constructed with primitive functions. These primitive functions have known derivatives.

We will use AD for Reverse Mode Differentiation. See previous post. (See how to link)
The idea is to extend the computational graph of $f$ with differentiation details, that are computed in the backward pass.

The computational cost is linear in the cost of evaluating the function. And the memory is consumption is proportional to the number of nodes in the computational graph.

In particular at most 4 d_0 cost of evaluating the functions. See Equation 4.21 in
A. Griewank and A. Walther. Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation. Second Edition. Society for Industrial and Applied Mathematics (SIAM), 2008.

Often people treat Autodiff as Autograd. But Autograd is the PyTorch implementation of Autodiff. See Trademark erosion
https://en.wikipedia.org/wiki/Generic_trademark#Trademark%20erosion

Give credit to lecture slides
https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf

CSC321 Lecture 10: Automatic Differentiation
CS.Toronto.edu University of Toronto
Roger Grosse

Sections:
- Recall what reverse mode differentiation is
- What do we mean by computational graph. And how do extend it.

https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
Use figures from here
Overview of PyTorch Autograd Engine
June 08 2021
Preferred Networks

Then going to do
- Example with one layer simple full connected neural network.
- Use gradients of functions you need it (less memory)
- Checkpointing

## Background

### Reverse Mode Differentiation
Link to other post. Phrase in terms of loss function.

{% include admonition.html type="info" title="Backward Mode Recursion" body="
Given functions $f_{0}(x) := x$ and $f_{1}, \dots, f_{m}$ with known Jacobians, we want to compute the Jacobian of their composition
$$
f(x) = f_{m} \circ \cdots \circ f_{1} (x),\quad
f_{i}:\mathbb{R}^{d_{i-1}} \to \mathbb{R}^{d_{i}}.
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
= \prod_{i=1}^{m} \frac{df_{i}}{df_{i-1}}.
$$
For $i = 0, \dots, m-1$,
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
}.
$$
"
%}

### Computational Graph
It's a Directed Acyclic Graph (DAG). https://en.wikipedia.org/wiki/Directed_acyclic_graph

Basically you have inputs pointing to a node representing the function.
Which points to / contain the value

<figure>
    <img src="/assets/figures/logxy-Computational-Graph.png"
         alt="logxy-Computational-Graph"
         />
    <figcaption>Figure 1: Computational graph. From reference pytorch blog.</figcaption>
</figure>

Get a graph. Give the Wikipedia page here


## Automatic Differentiation
A primitive function is any function whose derivative we predefine.

We now just need to be systematic in how we divide it by the primitives.

This may be the multiplication. In which case we define a Backward version of it.
This will take in the adjoint $\partial L / \partial v$ and the inputs.

<figure>
    <img src="/assets/figures/logxy-Extended-Computational-Graph.png"
         alt="logxy-Extended-Computational-Graph"
         />
    <figcaption>Figure 2: Extended Computational graph. From reference pytorch blog.</figcaption>
</figure>

As for a basic understanding of Autograd, this is sufficient for a basic understanding. The great thing about this is that it is very modular. The magic happens when you combine these together in mass to compute the derivative of a complicated thing. Divide and conquer.

Note that for efficiency reasons we do not have to compute the jacobian explicilty. For example consider a component wise application of an activation. This is a diagonal matrix.

Some people have done smart things to reduce memory consumption - which is an issue with reverse Autodiff. See the below.

### Primitive Functions
What you define as primitive functions has an impact on the memory requirements. Autodiff needs to determine all the intermediate values and store them.

Suppose we have function which is a matrix that is 1x4. If we implement `A@x` naively i.e. from the sum and compute the grad we get a more complicated computation graph which requires us to store more intermediate values. Here it would be $m(n + n-1)$. Products and the partial sums.

<figure>
    <img src="/assets/figures/Naive-MM-AD.png"
         alt="Naive-MM-AD"
         />
    <figcaption>Figure 3: Naive implementation of MM</figcaption>
</figure>

Whereas if we have function which tells us that the gradient is A then autograd will know that it does not have to store the extra states and reduce memory consumption.

There's also something with inplace operations.

<figure>
    <img src="/assets/figures/MM-AD.png"
         alt="MM-AD"
         />
    <figcaption>Figure 4: Better Implementation of MM</figcaption>
</figure>

In general if you know the derivative of a function, and simple to implement then use it.

You just need to define backward your `torch.autograd.Function`. And I think this will be one node on the computation graph.
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

Need to do more research here:
https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/

And read the other two things.

### Checkpointing

Consider the very simple case of function composition. So linear structure.
- Then if you store everything. O(n) meory and no extra recomputation.
- If you store nothing then O(1) memory but O(n^2) computation.
- Whereas if you store every k points, you only need to compute from that point on.

This has a memory cost of O(n/k + k). This is minimised by k = sqrt{n} leading to a memory cost of O(sqrt{n}).

And the extra recomputation is O(n) as values get recomputed at most once.

Take pictures from here.
https://github.com/cybertronai/gradient-checkpointing

TODO: You can contribute something here.
Still don't fully understand this.

The idea is that the cost is that the number of segments you have is.

This results in log(n) memory with O(n log n) computation.

https://arxiv.org/pdf/1604.06174.pdf

But ultimately as the memory cost of each layer is not the same, need to do different allocation. Paper explains more.

In particular, I think this makes the log(n) algorithm hard to implement in practice. So stick with sqrt n one.

#### PyTorch explains more
Something interesting here with the
https://pytorch.org/docs/stable/checkpoint.html

How they do it.
https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d

## Conclusion
This is just the overview. There are more details to optimise autograd. Find a reference.
https://arxiv.org/abs/1811.05031

A review of automatic differentiation and its efficient implementation
12 Nov 2018
Charles C Margossian.

If you have an explicit symbolic formula for the derivative of your function, better off just using that.

Otherwise, if working with something complicated like a neural network then best to do autograd.




