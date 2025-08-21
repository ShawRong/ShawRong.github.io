---
title: "Group Theory Note"
date: 2025-08-21T08:42:27.767Z
draft: false
tags: []
---

#grouptheory #math #note #lecture

1.1 Symmetry
===============

Naive concept about group: set of element with symmetrical operation.

1.2 Definition of Group
============

A group is a set with operation:
- $G\times G \to G$ i.e. $g_1 \cdot g_2 \to g_1g_2$ 

Axioms:
- $\forall g_1, g_2, g_3 \in G$, $(g_1 g_2) g_3 = g_1(g_2 g_3)$ (associativity)
- $\exists e \in G$, $g \cdot e = e \cdot g = g$ (identity)
- $\forall g \in G$, $\exists g' \in G$, where $g'g = e$, denote $g'$ as $g^{-1}$  (inverse)

About triangle, we can define our group as:
G: {e, R, R^2, S_A, S_B, S_C}

take it as example, we can see the following:
- $S_A^{-1} = S_A$
- $S_B^{-1} = S_B$
- $S_C^{-1} = S_C$
- $R^{-1} = R^2$
- $(R^2)^{-1} = R2$

1.3 Corollaries of the axioms
===========
- $e_1 = e_1e_2 = e_2$ (unique unit)
- $g'g = e$, $g''g = e$, $g''g'g = g''e = g''$, $g'g''g = g'e = g'$, $g' = g''$(unique inverse)
- $\forall g \in G$, $(g^{-1})^{-1} = g$
- $\forall g_1, g_2 \in G$, $g_1h = g_2h \implies g_1 = g_2$
- $xg_1 = g_2$ and $g_1 x = g_2$ have unique solution, $g_2g_1^{-1}$ 

Abellain
------------
Abellian: A group is called "Abellian", if $\forall g_1, g_2 \in G$, we have $g_1g_2 = g_2g_1$
Example:
- (Q, +), (R, +)
- (C, +), (F, +), (V, +) (C, F, V stand for complex number, field, vector space respectively)
- the multiplication on Q, C, R, F without 0.

They are abellian(g1g2 = g2g1)

The cyclic Group, $\mathbb{Z}/n\mathbb{Z}$ is abellian, too.

1.5 Order, Subgroup, Isomorphism
===========
Order of Group
------------
We denote the order of G, as |G|, which is the number  of operation in G.
$|\mathbb{S}_n| = n!$, $|\mathbb{G}| \in \mathbb{N} \cup \inf$  
we say the order of G can be identified as either finite or infinite.

Order of Element:
An order 'k' of element is the smallest $k \in \mathbb{N}$, s.t.  $g^k = e$, e.g. 
$e, g, g^2, g^3,g^4,\cdots,e$
$\text{Ord}\{g\} \in \mathbb{N} \cup {\inf}$

In Z, ord(0) = 1, Ord(k) = inf, ...

Subgroup
-------------
Def: G is a group, H is subset of G, H is a subgroup, if
- Closed, $\forall h_1, h_2 \in H$, $h_1 h_2 \in H$
- $e \in H$
- $\forall h \in H$, $h^{-1} \in H$.
Operation of H is the same as G.

Some corollaries:
- $H_1, H_2$ is subgroup of G, implies $H_1 \cap H_2$ is subgroup, too.
- $\forall G$, G and {e} are subgroup of G.

Example of Subgroups:
- G = Z, H is even Number.
- G = Z/10 Z, H = {0, 5}
- S, H = {e, (1 2 3, 2 3 1), (1 2 3, 3 1 2)}

Isomorphism
----------
f: G_1 -> g_2
f is isomorphism, if $f(g_1g_2) = f(g_1) f(g_2)$, here $g_1 g_2 \in G_1$, $g_1, g_2 \in G_2$.
And f is bijection.

- Composition of isomorphism is an isomorphism.
- Inverse of an isomorphism is also an isomorphism.

Lagrange Theorem
================
Theorem: Let G be a finite group, for $a \in G$, we have $a^{|G|} = e$.

|G| is the order of G, which stands for the number of elements in G.
Moreover, |G| can be divided by ord(a).