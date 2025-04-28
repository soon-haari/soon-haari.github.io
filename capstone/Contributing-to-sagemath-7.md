---
layout: post
category: capstone
title: "Contributing to SageMath (팀 \"이태관\"(중요))"
permalink: c7
---

For capstone class(2025/5/2).

## Goals
1. Fix XGCD implementations to use PARI
2. Fix `inverse_mod`, so that it calls appropriate XGCD
3. Profit

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Goal 1. Fix XGCD implementations

`inverse_mod` function doesn’t directly calls XGCD functions.
However, the goal is to first make a properly working Half-gcd XGCD function.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### FLINT vs NTL

```python
p = next_prime(2^1024)
P.<x> = PolynomialRing(GF(p), implementation='FLINT')
```

```sh
$ sage xgcdtest.sage
Traceback (most recent call last):
  File "/home/soon_haari/mysage/xgcdtest.sage.py", line 10, in <module>
    P = PolynomialRing(GF(p), implementation='FLINT', names=('x',)); (x,) = P._first_ngens(1)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring_constructor.py", line 728, in PolynomialRing
    return _single_variate(base_ring, names, **kwds)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring_constructor.py", line 796, in _single_variate
    implementation_names = specialized._implementation_names_impl(implementation, base_ring, sparse)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring.py", line 3524, in _implementation_names_impl
    raise ValueError("FLINT does not support modulus %s" % modulus)
ValueError: FLINT does not support modulus 179769313486231590772930519078902473361797697894230657273430081157732675805500963132708477322407536021120113879871393357658789768814416622492847430639474124377767893424865485276302219601246094119453082952085005768838150682342462881473913110540827237163350510684586298239947245938479716304835356329624224137859
```

- **FLINT** is fast, but only for small moduli.
- **NTL** is relatively slow, but can hold unlimited size of integers theoretically.


<br>

- Note: FLINT only supports moduli in `range(2^63)`
```python
sage: PolynomialRing(Zmod(2^63 - 1), implementation='FLINT', names='x')
Univariate Polynomial Ring in x over Ring of integers modulo 9223372036854775807
sage: PolynomialRing(Zmod(2^63), implementation='FLINT', names='x')
...
ValueError: FLINT does not support modulus 9223372036854775808
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Base format code

```python
import time

P.<x> = PolynomialRing(?)
deg = ?

a = P.random_element(deg)
b = P.random_element(deg)

st = time.time()
a.gcd(b)
en = time.time()
print(f"GCD took {(en - st):.2f}s.")

st = time.time()
a.xgcd(b)
en = time.time()
print(f"XGCD took {(en - st):.2f}s.")

st = time.time()
a._pari_with_name().gcd(b._pari_with_name())
en = time.time()
print(f"PARI GCD took {(en - st):.2f}s.")

st = time.time()
a._pari_with_name().gcdext(b._pari_with_name())
en = time.time()
print(f"PARI XGCD took {(en - st):.2f}s.")
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>


### 1. Small prime modulus with FLINT

```python
p = next_prime(2^16)
P.<x> = PolynomialRing(GF(p), implementation='FLINT')
deg = 50000
...
```
```
GCD took 0.19s.
XGCD took 0.30s.
PARI GCD took 0.64s.
PARI XGCD took 0.68s.
```

FLINT's GCD/XGCD both use half-gcd algorithm, and is faster than using PARI.
- Unimprovable with PARI.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>


### 2. Small prime modulus with NTL

```python
p = next_prime(2^16)
P.<x> = PolynomialRing(GF(p), implementation='NTL')
deg = 10000
...
```
```
GCD took 0.12s.
XGCD took 11.95s.
PARI GCD took 0.08s.
PARI XGCD took 0.09s.
```

NTL's GCD use half-gcd algorithm, however, slightly slower than PARI.

More tests:
```
# degree = 100000
GCD took 1.79s.
PARI GCD took 1.24s.
```

```
# degree = 1000000
GCD took 20.62s.
PARI GCD took 18.81s.
```

```
# degree = 100, but 10000 iterations
GCD took 0.12s.
PARI GCD took 0.50s.
```

For high degree, PARI is slightly faster, but for low degree, current implementation is faster.

<br>

NTL's XGCD doesn't use half-gcd algorithm, thus incredibly slower than PARI.
However, for low degree such as 10, still current implementation is faster than PARI.
```
XGCD took 0.33s.
PARI XGCD took 0.55s.
```

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>


### 3. Small composite modulus with FLINT

```python
n = 1073741827 * 1073741831
P.<x> = PolynomialRing(Zmod(n), implementation='FLINT')
deg = 100000
```

```
GCD took 1.66s.
XGCD took 2.57s.
PARI GCD took 3.58s.
PARI XGCD took 3.86s.
```

Conclusion: FLINT is completely unimprovable with PARI.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### 4. Multi-dimensional fields with small modulus
```python
P.<x> = PolynomialRing(GF(256), implementation='FLINT')
deg = 30000
```
```sh
$ sage xgcdtest.sage
Traceback (most recent call last):
  File "/home/soon_haari/mysage/xgcdtest.sage.py", line 9, in <module>
    P = PolynomialRing(GF(_sage_const_256 ), implementation='FLINT', names=('x',)); (x,) = P._first_ngens(1)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring_constructor.py", line 728, in PolynomialRing
    return _single_variate(base_ring, names, **kwds)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring_constructor.py", line 819, in _single_variate
    implementation_names = constructor._implementation_names(implementation, base_ring, sparse)
  File "/home/soon_haari/mysage/sage/src/sage/rings/polynomial/polynomial_ring.py", line 548, in _implementation_names
    raise ValueError("unknown implementation %r for %s polynomial rings over %r" %
            (implementation, "sparse" if sparse else "dense", base_ring))
ValueError: unknown implementation 'FLINT' for dense polynomial rings over Finite Field in z8 of size 2^8
```
FLINT doesn't support Multi-dimensional extension fields.

<br>

```python
P.<x> = PolynomialRing(GF(256), implementation='NTL')
deg = 30000
```
```
GCD took 27.59s.
XGCD took 31.42s.
PARI GCD took 7.29s.
PARI XGCD took 8.46s.
```

For degree 60000:
```
PARI GCD took 15.47s.
PARI XGCD took 18.73s.
```

PARI still use half-gcd algorithm, however NTL doesn't use half-gcd for both GCD and XGCD.
- NTL only supports half-gcd for **Prime modulus' GCD**

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### 5. Big prime modulus with NTL
```python
p = next_prime(2^1024)
P.<x> = PolynomialRing(GF(p), implementation='NTL')
deg = 5000
```
```
GCD took 0.79s.
XGCD took 32.75s.
PARI GCD took 2.12s.
PARI XGCD took 2.30s.
```

As expected, NTL's GCD is similar to PARI, but XGCD still very slow compared to PARI.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### 6. Big composite modulus with NTL

```python
p = next_prime(2^512)
q = next_prime(p)
P.<x> = PolynomialRing(Zmod(p * q), implementation='NTL')
deg = 5000
```

```
...
NotImplementedError: Ring of integers modulo 179...211 does not provide a gcd implementation for univariate polynomials
...
NotImplementedError: Ring of integers modulo 179...211 does not provide an xgcd implementation for univariate polynomials
PARI GCD took 2.12s.
PARI XGCD took 2.31s.
```

For NTL, both GCD and XGCD is not implemented, and PARI speed is equal to prime modulus PARI GCD/XGCD.

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

### Conclusion

- **FLINT** is fine.
- **NTL**'x XGCD for prime modulus can be improved using PARI.
- **NTL**'x GCD/XGCD isn't implemented and can be written using PARI.

Next week plans:
- Find NTL's GCD/XGCD for composite Zmod, and write my new implementations with PARI.
- ~~And maybe make a PR...~~

<img src="../../files/sagemath/line.png" width="800"/>

<br><br>

<img src="../../files/sagemath/thanku.png" width="800"/>