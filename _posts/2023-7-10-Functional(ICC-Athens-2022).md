---
layout: post
category: studies
---

<br><br>

I will review a challenge in category ["CTF Archive"](https://cryptohack.org/challenges/ctf-archive/) from [cryptohack.org](https://cryptohack.org/).

I didn't categorize this post as "write-up" because I wanted "write-up" category to be filled with only CTFs I participated.

<br>

### Description
- It only took me four heat deaths of the universe to encrypt this flag.

### functional.sage

```python
import numpy as np
from secret_params import (COEFFS,
                            ITERS # ITERS = secrets.randrange(13**37)
                            )
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib

F = GF(2^142 - 111)

def f(n):
    if n < len(COEFFS):
        return F(int(10000*np.sin(n)))
    return COEFFS.dot(list(map(f, n - 1 - np.arange(len(COEFFS)))))

def g(n):
    if n < 6:
        return F(int(10000*np.sin(len(COEFFS) + n)))
    return np.array([F(int(10000*np.log10(2 + i))) for i in range(6)]).dot([g(n - 6), h(n - 2), i(n - 3), g(n - 3), h(n - 4), i(n)]) + 2*n**3 + 42

def h(n):
    if n < 3:
        return F(int(10000*np.sin(len(COEFFS) + 6 + n)))
    return np.array([F(int(10000*np.log10(1337 + i))) for i in range(4)]).dot([h(n - 3), i(n - 1), g(n - 2), h(n - 1)]) + n

def i(n):
    if n < 3:
        return F(int(10000*np.sin(len(COEFFS) + 9 + n)))
    return np.array([F(int(10000*np.log10(31337 + i))) for i in range(5)]).dot([i(n - 2), g(n - 3), h(n - 3), h(n - 1), i(n - 1)]) + 1

def j(n):
    if n < 10^4:
        return F(sum(S3[d] for d in ZZ(n).digits(1337)))
    return np.array([F(int(10000*np.log(31337 + i))) for i in range(100)]).dot(list(map(j, n - 10^4 + 100 - np.arange(100))))

if __name__ == "__main__":
    print("Stage 1:")
    print("========")
    print([f(ITERS + k) for k in range(500)])

    print("Stage 2:")
    print("========")
    S3 = [i(ITERS + k) for k in range(1337)]

    print("Stage 3:")
    print("========")
    key = hashlib.sha256(str(j(ITERS)).encode()).digest()
    cipher = AES.new(key, AES.MODE_ECB)
    with open("flag.txt", "rb") as f:
        print(cipher.encrypt(pad(f.read().strip(), 16)).hex())
```

### output.txt
```
Stage 1:
========
[3938419846976491177353386476959871391250276, 
...(REDACTED, There's more of course), 
644082098542529088290239024585910512264303]
Stage 2:
========
Stage 3:
========
bf7d4897735f758539188cf654e6003ce5ebff7cbfd3ba766b653366435e53e73013713fae33cfc240e04d6a8122db42c3dd29a13d68b9c4ae7f314664f43703
```

The source code is pretty short, but the depth behind this definitely surprised me.

<br><br>

## Step 1. Recovering ITERS from given output

Gladly, the steps of this challenge are nicely separated.

As there are a lot of functions in this challenge, but to get ITERS, we only need to observe function `f`.

```python
F = GF(2^142 - 111)

def f(n):
    if n < len(COEFFS):
        return F(int(10000*np.sin(n)))
    return COEFFS.dot(list(map(f, n - 1 - np.arange(len(COEFFS)))))

if __name__ == "__main__":
    print("Stage 1:")
    print("========")
    print([f(ITERS + k) for k in range(500)])
```

When we observe function `f`, we can see that it is a function defined as:

- len(COEFFS) initial values for f(0) ~ f(len(COEFFS) - 1)
- if n >= len(COEFFS), dot product with the given COEFFS and f(n - 1) ~ f(n - len(COEFFS))

We can see that it is a basic `recurrence relation` like the Fibonacci sequence.

(P.S. The author said this operation took 4 [heat deaths of the universe](https://en.wikipedia.org/wiki/Heat_death_of_the_universe) assuming this operation takes O(N) time complexity, but when we look at it deeper, there's no memoization, so this operation takes depth of N recursive functions and O(2^N) time complexity haha.

I don't think O(2^2^128) will be done in ONLY 4 heat deaths of the universe.)

<br><br>

It is well known that we can calculate Fibonacci(N) with O(logN) time complexity by converting it to a matrix.

<center>$\begin{bmatrix} F_{n}  \\ F_{n + 1}\end{bmatrix} = \begin{bmatrix}0 & 1 \\ 1 & 1\end{bmatrix}\begin{bmatrix}F_{n - 1}\\F_{n}\end{bmatrix}$</center>

Because we can compute power of middle matrix with log time complexity.

With the same method, we can construct the matrix in function `f`'s case too.

<br>

But first, we MUST know the length of COEFFS, or we can't proceed anything. 

This can be done easily by using 500 outputs, and iterating the lengths.

Sage's solve_right can be used here checking the recurrence relation. If there are no root of COEFFS, the length is wrong.

<br>

```python
data = [3938419846976491177353386476959871391250276, ..., 644082098542529088290239024585910512264303]

assert len(data) == 500

p = 2^142 - 111
GFp = GF(2^142 - 111)

for l in range(1, 250):
	# l = length of COEFFS
	mat = []
	res = []

	for i in range(500 - l):
		mat.append(data[i:i + l])
		res.append(data[i + l])

	mat = Matrix(GFp, mat)
	res = vector(GFp, res)

	try:
		COEFFS = mat.solve_right(res)

		print(f"{len(COEFFS) = }")
		print(f"{COEFFS = }")

	except:
		pass
```

```
len(COEFFS) = 20
COEFFS = (75853883610309597642200581281276453016977, ..., 2499633494185931196982831794661758600052741)
len(COEFFS) = 21
COEFFS = (1955817776487665191870045349448891442406234, ..., 1958657366602383996322663779666409703587544, 0)
...
```


Note that there always exists a root for `l >= 20` followed by (l - 20) zero values, because obviously if we can express a value with 20 variables, we can definitely do it with more.

So we can assume len(COEFFS) is 20. 

<br>

Although I wish there was something in the challenge to prove that COEFFS cannot be reduced anymore.

Because `f`'s initial values completely depend on length of COEFFS. And 21 length COEFFS including 0 value can actually work.

I first thought `assert prod(COEFFS)` would be good enough, but realized it is not.

Not really sure of a nice way to solve this.

<br><br><br>

Now we know all the informations needed to make the recurrence relation.

And we can construct the matrix equation according to that relation:

<center>$\begin{bmatrix}
f_{n - 18} \\
f_{n - 17} \\
\vdots  \\
f_{n} \\
f_{n + 1}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & \cdots  & 0 & 0 \\
0 & 0 & \cdots & 0 & 0 \\
\vdots  & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 0 & 1 \\
CF_{0} & CF_{1} & \cdots & CF_{18} & CF_{19}
\end{bmatrix}
\begin{bmatrix}
f_{n - 19} \\
f_{n - 18} \\
\vdots  \\
f_{n - 1} \\
f_{n}
\end{bmatrix}$</center>

We have the initial vector, and vector after ITERS multiplication, so only DLP problem left to get ITERS.

<br>

Gladly, there is a challenge called "The Matrix Revolutions" from cryptohack's [Diffie-Hellman](https://cryptohack.org/challenges/diffie-hellman/) category.

I reviewed the challenge and solutions, and especially got help from hellman's solution using `charpoly()`.

But DLP 











