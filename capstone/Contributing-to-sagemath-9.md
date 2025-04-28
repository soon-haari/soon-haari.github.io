---
layout: post
category: capstone
title: "Contributing to SageMath (팀 \"이태관\"(중요))"
permalink: /c9/index
---

For capstone class(2025/5/16).

<br>

While PR and issues for GCD/XGCD is cooking...

<img src="../../files/sagemath/cooking.webp" width="250"/>

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Matrix's solve_right

```python
sage: M = random_matrix(GF(7), 10)
sage: v = random_vector(GF(7), 10)
sage: M
[0 2 0 2 3 4 3 0 0 1]
[5 0 3 5 6 5 3 2 6 3]
[6 2 1 0 3 6 5 2 4 2]
[6 1 3 6 2 6 2 2 3 4]
[2 1 4 0 1 1 3 0 0 1]
[0 5 6 1 0 4 3 6 2 5]
[1 6 1 0 1 6 2 5 5 5]
[1 1 2 5 0 0 0 2 1 4]
[6 5 3 0 2 5 6 5 4 5]
[5 5 5 5 6 5 5 4 5 5]
sage: v
(0, 0, 4, 3, 6, 0, 4, 3, 3, 0)
sage: M.solve_right(v)
(5, 5, 2, 5, 3, 4, 3, 2, 1, 2)
sage: M * M.solve_right(v) == v
True
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Case when roots are not unique

```python
sage: M
[2 6 4 2 1 4 2 2 4 5]
[6 5 5 1 0 5 0 4 5 2]
[4 4 3 2 3 1 4 6 4 3]
[6 3 1 5 6 5 4 0 0 5]
[3 6 6 0 1 5 6 2 1 3]
[5 1 2 3 2 3 1 1 3 0]
[6 0 6 1 1 6 2 0 6 2]
[0 2 4 3 3 4 3 0 5 2]
[1 2 0 1 2 1 2 2 2 5]
[6 0 2 3 5 5 3 3 4 4]
sage: v
(4, 6, 2, 3, 4, 2, 0, 4, 6, 0)
sage: r
(1, 1, 0, 2, 2, 4, 6, 2, 2, 6)
sage: a = M.solve_right(v)
sage: M * a
(4, 6, 2, 3, 4, 2, 0, 4, 6, 0)
sage: M * (a + r)
(4, 6, 2, 3, 4, 2, 0, 4, 6, 0)
sage: M * (a + r * 2)
(4, 6, 2, 3, 4, 2, 0, 4, 6, 0)
```

<br>

#### Reason for this?

When `M` has a right nullspace vector, then `M * r = 0` for such vector `r`.

- Can be calculated with `right_kernel` or `right_kernel_matrix`.

```python
sage: M.right_kernel_matrix()
[1 1 0 2 2 4 6 2 2 6]
sage: M * M.right_kernel_matrix().T
[0]
[0]
[0]
[0]
[0]
[0]
[0]
[0]
[0]
[0]
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### All roots' form

Note: This property only holds if the base ring is a field, if the base ring is not a field, RREF isn't even properly defined, and loses lots of property.

<br>

Assuming a valid root is `r`, then all roots has form of `r + a` where `a` is an element of vector space of right kernel.

How to iterate all roots:
```python
M = random_matrix(GF(7), 10, 15)
v = random_vector(GF(7), 10)

from itertools import product

def iterate_all(r, ker):
    l = ker.nrows()
    P = r.base_ring()

    for it in product(P, repeat=l):
        yield r + sum(a * b for a, b in zip(ker, it))

def all_roots_with_total(M, v):
    try:
        r = M.solve_right(v)
    except:
        return iter([]), 0

    ker = M.right_kernel_matrix()

    l = ker.nrows()
    P = r.base_ring()

    tot = P.order()^l

    return iterate_all(r, ker), tot

it, total = all_roots_with_total(M, v)
for i in range(total):
    assert M * next(it) == v
```

Disadvantage:
- It internally calls 2 Matrix reductions, inefficient
- If root doesn't exist, try-except is also inefficient

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### One shot kernel!

Solving by appending the target vector to the matrix, then calculating right kernel once.

> Solves both of the disadvantage mentioned above.

```python
M = random_matrix(GF(7), 10, 15)
v = random_vector(GF(7), 10)

from itertools import product

def iterate_all(r, ker):
    l = ker.nrows()
    P = r.base_ring()

    for it in product(P, repeat=l):
        yield r + sum(a * b for a, b in zip(ker, it))

def all_roots_with_total(M, v):
    P = v.base_ring()

    ker = block_matrix([[-v.column(), M]]).right_kernel_matrix()

    if ker[0, 0] == P(0):
        return iter([]), 0

    assert ker[0, 0] == P(1)
    r, ker = ker[0][1:], ker[1:, 1:]

    l = ker.nrows()

    tot = P.order()^l

    return iterate_all(r, ker), tot

it, total = all_roots_with_total(M, v)
for i in range(total):
    assert M * next(it) == v
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### P.S.

1. I thought of this method during a CTF.<br><br>
<img src="../../files/sagemath/miko.png" width="300"/>
2. There was a worthy conversation, that triggered me to actually try suggesting to SageMath.
<img src="../../files/sagemath/convo.png" width="800"/>

<img src="../../files/sagemath/line.png" width="800"/>
<br><br>

<img src="../../files/sagemath/thanku.png" width="800"/>