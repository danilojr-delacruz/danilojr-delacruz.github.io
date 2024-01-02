---
title: Automatic Differentiation
date: 2024-01-02
categories: [Machine Learning, Implementation]
tags: [ml]     # TAG names should always be lowercase
publish: true
math: true
---

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

### Computational Graph
It's a Directed Acyclic Graph (DAG).

Get a graph. Give the Wikipedia page here


## Extending the Computational Graph

## Primitive Functions

## Checkpointing

Give simple explanation

I don't understand how they get the log.
https://github.com/cybertronai/gradient-checkpointing

https://arxiv.org/pdf/1604.06174.pdf

Something interesting here with the
https://pytorch.org/docs/stable/checkpoint.html

How they do it.
https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d

## Conclusion
This is just the overview. There are more details to optimise autograd. Find a reference.

If you have an explicit symbolic formula for the derivative of your function, better off just using that.

Otherwise, if working with something complicated like a neural network then best to do autograd.




