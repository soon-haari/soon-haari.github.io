---
layout: post
category: capstone
title: "Contributing to SageMath (팀 \"이태관\"(중요))"
permalink: /cTBD/index
---


### user202729's new reviews

<img src="../../files/sagemath/u20.png" width="800"/>

<img src="../../files/sagemath/u21.png" width="800"/>

<img src="../../files/sagemath/u22.png" width="800"/>

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Asking for help...

<img src="../../files/sagemath/chat.png" width="800"/>

[https://doc.sagemath.org/html/en/developer/coding_basics.html#running-automated-doctests](https://doc.sagemath.org/html/en/developer/coding_basics.html#running-automated-doctests)


<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Fix and doctests

```
.. NOTE::

    Algorithm is set to 'pari' in default, but user may set it to 'ntl'
    to use the algorithm using ``ntl_ZZ_pX.gcd``. Algorithm 'pari' implements
    half-gcd algorithm in some cases, which is significantly faster
    for high degree polynomials. Generally without half-gcd algorithm, it is
    infeasible to calculate gcd/xgcd of two degree 50000 polynomials in a minute
    but TEST shows it is doable with algorithm 'pari'.

...
```

```python
TESTS::

    sage: P.<x> = PolynomialRing(GF(next_prime(2^512)), implementation='NTL')
    sage: degree = 50000
    sage: g_deg = 10000
    sage: g = P.random_element(g_deg).monic()
    sage: a, b = P.random_element(degree), P.random_element(degree)
    sage: r, s, t = a.xgcd(b)
    sage: (r == a.gcd(b)) and (r == s * a + t * b)
    True
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

<img src="../../files/sagemath/thanku.png" width="800"/>