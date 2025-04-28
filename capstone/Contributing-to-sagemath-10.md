---
layout: post
category: capstone
title: "Contributing to SageMath (팀 \"이태관\"(중요))"
permalink: /c10
---

For capstone class(2025/5/23).

## Goals
1. Fix XGCD implementations to use PARI
2. Fix `inverse_mod`, so that it calls appropriate XGCD
3. Profit

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Goal 1. Fix XGCD implementations

#### Last week's Result
- Made issue: [https://github.com/sagemath/sage/issues/40016](https://github.com/sagemath/sage/issues/40016)
- Made PR: [https://github.com/sagemath/sage/pull/40017](https://github.com/sagemath/sage/pull/40017)
- Waiting for approval / review

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### user202729's 4 review

<img src="../../files/sagemath/user202729_1.png" width="800"/>

- Suggested fix: change error message start with lowercase character.

<br>

<img src="../../files/sagemath/user202729_2.png" width="800"/>

- Suggested fix: use `pari(a)` instead of `a._pari_with_name()`, however `pari(a)` has some errors so might as well keep current version.

<br>

<img src="../../files/sagemath/user202729_3.png" width="800"/>

- Suggested fix: another way of error handling.

<br>

<img src="../../files/sagemath/user202729_4.png" width="800"/>

- Suggested fix: adding `algorithm=?` parameter, and make it selectable.

<img src="../../files/sagemath/line.png" width="800"/>

### According fix

`sage/rings/polynomial/polynomial_modn_dense_ntl.pyx`
Original:
```python
    P = self.parent()
    s, t, r = self._pari_with_name().gcdext(other._pari_with_name())
    s = P(s)
    t = P(t)
    r = P(r)
    c = r.leading_coefficient()
    if c != P(0):
        s /= c
        t /= c
        r /= c
    return r, s, t
```

New:
```python
@coerce_binop
def xgcd(self, other, algorithm='pari'):
    ...
    if algorithm not in ['pari', 'ntl']:
        raise ValueError(f"unknown implementation %r for xgcd function over %s" % (algorithm, self.parent()))

    if algorithm == 'pari':
        P = self.parent()
        s, t, r = self._pari_with_name().gcdext(other._pari_with_name())
        s = P(s)
        t = P(t)
        r = P(r)
        c = r.leading_coefficient()
        if c != P(0):
            s /= c
            t /= c
            r /= c
        return r, s, t
    else:
        r, s, t = self.ntl_ZZ_pX().xgcd(other.ntl_ZZ_pX())
        return self.parent()(r, construct=True), self.parent()(s, construct=True), self.parent()(t, construct=True)
```

Doc update:
<img src="../../files/sagemath/modn_ntl_doc.png" width="800"/>

<br>

`sage/rings/polynomial/polynomial_element.pyx`

Original:
```python
    try:
        P = self.parent()
        g = self._pari_with_name().gcd(other._pari_with_name())
        g = P(g)
        c = g.leading_coefficient()
        if c != P(0):
            g /= c
        return g
    except (PariError, ZeroDivisionError):
        raise ValueError("Failed to calculate GCD on polynomials over composite ring.")
    except AttributeError:
        pass
    try:
        doit = self._parent._base._gcd_univariate_polynomial
    except AttributeError:
        raise NotImplementedError("%s does not provide a gcd implementation for univariate polynomials"%self._parent._base)
    else:
        return doit(self, other)
```

New:
```python
def gcd(self, other, algorithm='pari'):
    ...
    if algorithm not in ["pari", "generic"]:
        raise ValueError(f"unknown implementation %r for xgcd function over %s" % (algorithm, self.parent()))
    if algorithm == 'pari':
        P = self.parent()
        g = self._pari_with_name().gcd(other._pari_with_name())
        g = P(g)
        c = g.leading_coefficient()
        if c != P(0):
            g /= c
        return g

    try:
        doit = self._parent._base._gcd_univariate_polynomial
    except AttributeError:
        raise NotImplementedError("%s does not provide a gcd implementation for univariate polynomials"%self._parent._base)
    else:
        return doit(self, other)
```

Doc update:
<img src="../../files/sagemath/element_doc.png" width="800"/>

Same with `xgcd`.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Issue for inverse_mod (Goal 2)

<img src="../../files/sagemath/issue2.png" width="800"/>
<img src="../../files/sagemath/smirk.png" width="300"/>

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Plans (Again?)

- Wait.
- Goal 1 finish.
- Work on goal 2.
- Profit.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

<img src="../../files/sagemath/thanku.png" width="800"/>