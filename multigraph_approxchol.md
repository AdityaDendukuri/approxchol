# Multigraph Approximate Cholesky: Theory, Mechanics, and Analysis

---

## 1. Executive Summary

Approximate Gaussian Elimination for Graph Laplacians (**Kyng & Sachdeva, FOCS 2016**) provides an $O(m \operatorname{polylog} n)$ algorithm to compute an approximate sparse Cholesky factor $L L^T \approx A$. While asymptotically near-linear, the original single-sample variant (**AC1**) suffers in practice from high sampling variance, spectral distortion, and sensitive convergence.

The **Multigraph formulation** (**Gao, Kyng, & Spielman, SISC 2026**, also known as **AC2 / $\text{AC}(k)$**) resolves these issues by treating edges not as scalar weights, but as bundles of independent multi-edge copies. This document provides the mathematical foundation, proofs of unbiasedness, variance analysis, and implementation mechanics of the multigraph approach.

---

## 2. Classical Gaussian Elimination on Graph Laplacians

Let $G = (V, E, w)$ be a weighted, undirected graph on $n$ vertices with positive edge weights $w_{uv} > 0$. The graph Laplacian matrix $A \in \mathbb{R}^{n \times n}$ is defined as:

$$A = \sum_{\{u, v\} \in E} w_{uv} (\mathbf{e}_u - \mathbf{e}_v)(\mathbf{e}_u - \mathbf{e}_v)^T$$

where $\mathbf{e}_u \in \mathbb{R}^n$ is the standard basis vector.

### 2.1 The Exact Schur Complement (Dense Clique Formation)
When eliminating a pivot vertex $k$ with diagonal pivot $D_k = A_{kk} = \sum_{i \in \mathcal{N}_k} w_{ki}$, exact Gaussian elimination computes the rank-1 update on the remaining Schur complement:

$$S_{\text{exact}} = A - \frac{1}{D_k} A_{:, k} A_{k, :}$$

Graphically, eliminating vertex $k$ removes its star graph (all incident edges $\{k, i\}$) and replaces it with a **complete clique** among all neighbors $\mathcal{N}_k$. For every pair of neighbors $i, j \in \mathcal{N}_k$ ($i \ne j$), an edge is inserted with weight:

$$w_{ij}^{\text{exact}} = \frac{w_{ki} \cdot w_{kj}}{D_k}$$

$$\begin{array}{ccc}
\textbf{Before Elimination (Star)} & & \textbf{After Elimination (Clique)} \\
\begin{matrix}
& [k] & \\
\swarrow & \downarrow & \searrow \\
[1] & [2] & [3]
\end{matrix}
& \Longrightarrow &
\begin{matrix}
[1] & \longleftrightarrow & [2] \\
\updownarrow & \nwsearrow & \updownarrow \\
[3] & \longleftrightarrow & \dots
\end{matrix}
\end{array}$$

If $\deg(k) = d$, this step introduces $\binom{d}{2} = O(d^2)$ fill-in edges. Across all elimination steps, this leads to catastrophic fill-in ($O(n^2)$ nonzeros).

---

## 3. The Original Method (AC1) and Its Failure Modes

To avoid quadratic fill-in, **AC1** replaces the dense clique with a **random sparse subgraph** of only $O(d)$ edges that matches the clique in expectation.

### 3.1 AC1 Sampling Algorithm
1. Collect all neighbors of pivot $k$: $\mathcal{N}_k = \{1, 2, \dots, d\}$ with weights $w_i = -A_{ki}$.
2. Sort neighbors in ascending order of edge weight:
   $$w_1 \le w_2 \le \dots \le w_d$$
3. For each neighbor $i \in \{1, \dots, d-1\}$, define the **heavier tail** weight:
   $$R_i = \sum_{j = i + 1}^d w_j$$
4. Draw **exactly 1 partner** $j \in \{i+1, \dots, d\}$ with probability:
   $$\mathbb{P}(\text{select } j) = \frac{w_j}{R_i}$$
5. Insert a single sampled edge $(i, j)$ with rescaled weight:
   $$\widehat{w}_{ij} = \frac{w_i R_i}{D_k}$$

### 3.2 The Fundamental Bottleneck of AC1
While $\mathbb{E}[\widehat{w}_{ij}] = \frac{w_i w_j}{D_k}$ is mathematically unbiased, AC1 is an **"all-or-nothing" gamble**:
* Neighbor $i$ places **100% of its Schur complement mass** onto a single random partner $j$.
* If $w_i$ is large, edge $(i, j)$ becomes an artificially stiff link, while $i$ completely loses connection to the other $d-2$ neighbors in that step.
* **Spectral Consequence:** AC1 exhibits high sampling variance, frequently creating near-singular cuts or poorly connected clusters. The preconditioned condition number $\kappa(M^{-1} A)$ is often large ($\ge 100$) and highly sensitive to the random seed.

---

## 4. The Multigraph Formulation ($\text{AC}2$ / $\text{AC}(k)$)

The **Multigraph method** replaces the single-draw lottery by treating every edge as a collection of independent multi-edge copies (degrees of freedom).

```
Simple Edge (AC1):       u --------------------------------- v   (1 edge, weight w)

Multigraph Edge (AC2):   u ================================= v   (2 copies, weight w/2 each)
```

### 4.1 Edge Representation
Every edge $e = \{u, v\}$ is characterized by two quantities:
1. **Total Weight** $w_{uv} \in \mathbb{R}^+$
2. **Multiplicity / Copy Count** $c_{uv} \in \mathbb{Z}^+$

Original edges in the graph start with multiplicity $c_{uv} = s$ (where $s = 2$ for AC2).

---

## 5. Mathematical Derivations and Proofs

### 5.1 Proof of Unbiasedness for Multigraph Sampling

Let neighbor $i$ have weight $w_i$, multiplicity $c_i$, and heavier tail sum $R_i = \sum_{j > i} w_j$.
The number of independent draws allocated to neighbor $i$ is:

$$r = \min(c_i, s)$$

Each of the $r$ draws is assigned weight:

$$\widehat{w} = \frac{w_i}{r} \cdot \frac{R_i}{D_k}$$

Let the random variable $I_{m, j} \in \{0, 1\}$ indicate whether the $m$-th draw selects partner $j$:

$$\mathbb{P}(I_{m, j} = 1) = \frac{w_j}{R_i}, \quad \text{for } j > i$$

The total weight $\widehat{W}_{ij}$ added between vertices $i$ and $j$ across all $r$ draws is:

$$\widehat{W}_{ij} = \sum_{m=1}^r I_{m, j} \cdot \widehat{w} = \widehat{w} \sum_{m=1}^r I_{m, j}$$

Taking the expectation:

$$\begin{aligned}
\mathbb{E}[\widehat{W}_{ij}] &= \mathbb{E}\left[ \widehat{w} \sum_{m=1}^r I_{m, j} \right] \\
&= \widehat{w} \sum_{m=1}^r \mathbb{E}[I_{m, j}] \\
&= \left( \frac{w_i}{r} \frac{R_i}{D_k} \right) \cdot \sum_{m=1}^r \left( \frac{w_j}{R_i} \right) \\
&= \left( \frac{w_i}{r} \frac{R_i}{D_k} \right) \cdot \left( r \cdot \frac{w_j}{R_i} \right) \\
&= \frac{w_i w_j}{D_k}
\end{aligned}$$

$$\boxed{\mathbb{E}[\widehat{W}_{ij}] = w_{ij}^{\text{exact}}}$$

$$\tag*{$\blacksquare$}$$

Thus, the multigraph sampling estimator is **strictly unbiased** for any choice of $r \ge 1$.

---

### 5.2 Variance Reduction Analysis

Consider the variance of the total weight added to pair $(i, j)$.

For a single draw ($r = 1$, AC1):

$$\operatorname{Var}(\widehat{W}_{ij}^{(1)}) = \mathbb{E}\left[(\widehat{W}_{ij}^{(1)})^2\right] - \left(\frac{w_i w_j}{D_k}\right)^2 = \left(\frac{w_i R_i}{D_k}\right)^2 \left(\frac{w_j}{R_i}\right) - \left(\frac{w_i w_j}{D_k}\right)^2$$

For $r$ independent draws with weight scaled by $1/r$ (AC2 with $r=2$):

$$\widehat{W}_{ij}^{(r)} = \frac{1}{r} \sum_{m=1}^r X_m, \quad \text{where } X_m = \left(\frac{w_i R_i}{D_k}\right) I_{m, j}$$

Since the draws are mutually independent:

$$\operatorname{Var}\left( \widehat{W}_{ij}^{(r)} \right) = \frac{1}{r^2} \sum_{m=1}^r \operatorname{Var}(X_m) = \frac{1}{r} \operatorname{Var}(X_1)$$

For $r = 2$ (AC2):

$$\boxed{\operatorname{Var}(\widehat{W}_{ij}^{(\text{AC2})}) = \frac{1}{2} \operatorname{Var}(\widehat{W}_{ij}^{(\text{AC1})})}$$

**Key Result:** Splitting original edges into 2 copies **halves the sampling variance** of every elimination step.

---

## 6. Dynamic Importance Sampling ("The Highway Effect")

A critical advantage of the multigraph formulation is its self-tuning capability during elimination:

$$\begin{matrix}
\textbf{Step 1:} & \text{Elimination of } u \implies \text{samples edge } (a, b) & \implies & c_{ab} = 1 \\
\textbf{Step 2:} & \text{Elimination of } v \implies \text{samples edge } (a, b) & \implies & c_{ab} = 1 + 1 = 2
\end{matrix}$$

```
Path 1 ───► (a) ════════ (b) ◄─── Path 2
            copies = 2 (reinforced highway)
```

1. When multiple random elimination paths place fill-in onto the same edge $(a, b)$, the copy counter increments ($c_{ab} \leftarrow c_{ab} + 1$).
2. The multigraph implicitly identifies that $(a, b)$ is a **high-traffic structural highway**.
3. When vertex $a$ is eventually eliminated, having $c_{ab} \ge 2$ grants it $\min(c_{ab}, 2) = 2$ draws, splitting its weight across multiple paths and preserving the graph's global backbone.

---

## 7. Degree Queue in Multigraph Elimination

In standard Cholesky, the minimum-degree heuristic selects the vertex with the fewest distinct neighbors. In a multigraph, the priority metric must reflect total **multi-edge incident copies**:

$$\deg(u) = \sum_{v \in \mathcal{N}_u} c_{uv}$$

### Queue Operations:
* **Initialization:** $\deg(u) = \sum_{v \ne u} c_{uv}$
* **Edge Insertion:** $\text{add\_pair}(u, v, w, c) \implies \deg(u) \leftarrow \deg(u) + c, \quad \deg(v) \leftarrow \deg(v) + c$
* **Star Removal:** $\text{remove\_pivot}(k) \implies \deg(u) \leftarrow \deg(u) - c_{ku}$

This ensures elimination prioritizes nodes that are light in both connectivity and multi-edge density.

---

## 8. Comparative Synthesis

| Metric / Property | Exact Cholesky ($LDL^T$) | Single-Sample (AC1) | Multigraph (AC2) |
| :--- | :--- | :--- | :--- |
| **Edge Storage** | Scalar weight $w$ | Scalar weight $w$ | Tuple $\{w, \text{copies}\}$ |
| **Initial Multiplicity** | N/A | $1$ | $s = 2$ |
| **Draws per Neighbor** | All $d - 1 - i$ partners | $1$ draw | $\min(\text{copies}_i, 2)$ draws |
| **Expected Factor** | $G D G^T = P^T A P$ | $\mathbb{E}[G D G^T] = P^T A P$ | $\mathbb{E}[G D G^T] = P^T A P$ |
| **Factor Nonzeros $\text{nnz}(L)$** | $O(n^2)$ (dense fill-in) | $\approx 1.5 \times \operatorname{nnz}(A)$ | $\approx 2.0 \times \operatorname{nnz}(A)$ |
| **Per-Step Variance** | $0$ | $\sigma^2$ | $\frac{\sigma^2}{2}$ |
| **Preconditioner Condition $\kappa$** | $1.0$ (direct solve) | $50 - 300+$ (erratic) | $5 - 20$ (tight & stable) |
| **PCG Iteration Count** | $1$ iteration | $80 - 250$ iterations | **$15 - 35$ iterations** |

---

## 9. Conclusion

The multigraph formulation elevates Randomized Approximate Cholesky from a theoretical curiosity into a robust, industrial-grade linear solver. By simply tracking a small integer `copies` counter:
1. Variance is reduced by $50\%$ per step.
2. Accidental graph disconnections are practically eliminated.
3. Conjugate Gradient iteration counts drop by **$3\times$ to $10\times$** with negligible memory overhead.
