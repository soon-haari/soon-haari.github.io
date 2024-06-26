---
layout: post
category: studies
title: "Functional(2022 ICC Athens)"
---

<br><br>

I will review a challenge in category ["CTF Archive"](https://cryptohack.org/challenges/ctf-archive/) from [cryptohack.org](https://cryptohack.org/).

I didn't categorize this post as "write-up" because I wanted "write-up" category to be filled with only CTFs I participated.

This was a completely new type of challenge during my entire crypto path, and I really enjoyed it.

The entire challenge required deep knowledge of linear algebra, crypto, algorithms even.

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

<br><br><br>

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

Not really sure of a nice way to solve this problem.

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

Actually Matrix itself can be DLPed too, but I thought the generated order is too large compared to polynomial. Doesn't really matter though.

For k * k matrix on GF(p) I learned that the order is $(p^k - 1)(p^k - p)(p^k - p^2) ... (p^k - p^{k-1})$ from [ASIS CTF 2022](https://asisctf.com) - Vindica.

<br><br>

When I factored the polynomial from charpoly(), It resulted like this:
```
(x^2 + 2068268905390596798719245736993915846629200*x + 634339472864573949596883090530146493127425) * (x^2 + 4188499892935915887716683610622478376038780*x + 4289906949466513447764172549738291287262736) * (x^3 + 31337)^3 * (x^7 + 2393970306752867687349097994046027930269265*x^6 + 4240925555251609271003753212718693332030014*x^5 + 5481963693833288113484648617324833274476930*x^4 + 3950536897508533279095380034516993640063002*x^3 + 958182397373590305928885762825477291936086*x^2 + 595941645193337044819907057654877588408728*x + 2684083987109102269511552970105503551935140)
```

Since the degree of factors are: `2, 2, 3^3, 7`

We can assume that the order is: 

<center>$lcm(p^2 - 1, p^9-p^6, p^7 - 1)$</center>

And checked if it works. It worked well.

```python
tot_order = (p - 1) * (p + 1) * (p^2 + p + 1) * ((p^7 - 1) // (p - 1)) * p^6
assert pow(x, tot_order, poly) == 1
```

<br><br>

Now for DLP, Pohlig-Hellman needs small factors of the order. 

I tried factoring that `tot_order` and resulted there are only these reasonable factors.

```python
small_factors = 2^5 * 3^2 * 13 * 47 * 419 * 643 * 1249 * 888721 * 1406497 * 10936843 * 830370383
assert tot_order % small_factors == 0
```

Moreover, our base `x`'s order is more important than the Field's order.

Turns out `x` is already a multiple of `13, 643, 1249`, so we have to exclude those factors during the DLP operation.

```python
fcts = [2^5, 3^2, 47, 419, 888721, 1406497, 10936843, 830370383]
```

```python
sage: prod([2^5, 3^2, 47, 419, 888721, 1406497, 10936843, 830370383]).bit_length()
116
sage: (13**37).bit_length()
137
```

We can see the remainder from Pohlig-Hellman is still 21 bits behind from the required ITERS' size.

Gladly, 21 bits is Fair enough for Exhaustive search.

<br><br>

I didn't find any kind of sage built-in for polynomial logarithms.

I implemented it myself using baby-step, giant-step method.

<br>

After Pohlig-Hellman method, exhaustive search is required to get complete ITERS.

```python
    dlog = crt(rem, mod)
    mod = prod(mod)

    assert pow(x, largos * dlog, poly) == pow(Gmul_poly, largos, poly)


    add = pow(x, mod, poly)
    st = pow(x, dlog, poly)

    print(mod)

    for i in trange(2^23):
        if st == Gmul_poly:
            ans = dlog + mod * i
            break
        st *= add
        st %= poly

    print(ans)
    assert pow(x, ans, poly) == Gmul_poly
```

```
100%|████████████████████████████| 8/8 [00:12<00:00,  1.54s/it]
64383183900624973568315342058469152
 15%|█▉           | 1228897/8388608 [01:09<06:42, 17783.95it/s]
79120327624133200239720213852419346424887
```

We Found `ITERS = 79120327624133200239720213852419346424887` finally.

<br><br><br>

## Step 2. Calculating S3

```python
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



if __name__ == "__main__":
    ...
    print("Stage 2:")
    print("========")
    S3 = [i(ITERS + k) for k in range(1337)]
```

Now we have to generate list of S3 which is defined as:

`[i(ITERS + k) for k in range(1337)]`

ITERS' value is very large, so we have to find an efficient, especially O(logN) way to compute function `i`.

(I first thought this challenge was finished after I calculated ITERS. It was not.)

<br><br>

Function `g`, `h`, `i` are defined as:
- $g(n) = Cg(n - 6) + Ch(n - 2) + Ci(n - 3) + Cg(n - 3) + Ch(n - 4) + Ci(n)$
$+ 2n^3 + 42$
- $h(n) = Ch(n - 3) + Ci(n - 1) + Cg(n - 2) + Ch(n - 1) + n$
- $i(n) = Ci(n - 2) + Cg(n - 3) + Ch(n - 3) + Ch(n - 1) + Ci(n - 1) + 1$

I wrote every known constant value as C.

<br><br>

This time the goal is not the DLP, but just generalizing the function in any kind of way.

And with matrix construction, that can be actually easily done,

If it wasn't for additional $2n^3 + 42$, $n$, $1$.

<br><br>

Then how we can remove those disturbing additional tails?

Gladly, I have faced these kind of question during math solving before.

<br>

Let's give an example of a recurrence relation.

<center>$a_{n + 1} = 2a_{n} + 3$</center>

How can we generalize this?

We can sum up some constant number in both side to make the form equal.

<center>$a_{n + 1} + 3 = 2(a_{n} + 3)$</center>

Then we define $a'\_{n} = a_{n} + 3$ and $a'\_{n}$ can easily be generalized.

<br><br>

How about this case?

<center>$b_{n + 1} = 2b_{n} + n$</center>

Little bit trickier, but doable.

<center>$b_{n + 1} + n + 1 = 2b_{n} + n + n + 1$</center>

<center>$b_{n + 1} + (n + 1) = 2(b_{n} + n) + 1$</center>

<center>$b_{n + 1} + (n + 1) + 1 = 2(b_{n} + n + 1)$</center>

Finally, we can define $b'\_{n} = b_{n} + n + 1$, and $b'\_{n + 1} = 2b'\_{n}$ comes out as a result.

<br><br>

There are more exceptions like this, but everything is fine if we just extend the degrees of $n$s.

<center>$c_{n + 1} = c_{n} + n$</center>

This one is impossible with just constant, and 1 degree of $n$.

<center>$c_{n + 1} - \frac{(n+1)^2}{2} = c_{n} + n- \frac{(n+1)^2}{2} = c_{n} - \frac{n^2}{2} - \frac{1}{2}$</center>

<center>$c_{n + 1} - \frac{(n+1)^2}{2} + \frac{n + 1}{2} = c_{n} - \frac{n^2}{2} + \frac{n}{2}$</center>

<br>

<center>$c'_{n} = c_{n} - \frac{n^2}{2} + \frac{n}{2} = constant$</center>

<br><br>

So the conclusion is, by defining $g'\_{n} = g_{n} + dn^3 + cn^2 + bn + a$,

and `h`, `i` as well, we can remove all the polynomial tails of $n$.

(If that doesn't have a solution, we can always add $en^4$, but until $dn^3$ worked fine.)

We have 4 coefficients per function, so total 12 variables.

<br>

Because no any other variables are multiplied, and 2 or more coefficients are never multiplied to each other,

finding those coefficients is easy as solving 12 linear equations with 12 variables.

<br><br>

After that, when we finally have recurrence relations as perfectly linear equations,

we can construct matrix equations, but with a little sense.

<br>

Unlike step 1, we have 3 functions now.

So we just have to set the old vector with 

`g(n - 5 ~ n), h(n - 5 ~ n), i(n - 5 ~ n)`

and the new vector with 

`g(n - 4 ~ n + 1), h(n - 4 ~ n + 1), i(n - 4 ~ n + 1)`

Middle matrix has to be 18 * 18 of size.

<br>

By the way, `g(n)` including `i(n)` itself was very disturbing,
so I decided to put `i(n)`'s recurrence directly into `g(n)`.

Theory for Step 2. is already extremely difficult, but implementing this in code is just another level LOL.

<br>

Front two values of S3 resulted as:
```
S3: [3344857554155236898658903831873044678801896, 3649959069781203859142471843095989397157559]
```

<br><br><br>

## Step 3. Calculating j(n)

```python
def j(n):
    if n < 10^4:
        return F(sum(S3[d] for d in ZZ(n).digits(1337)))
    return np.array([F(int(10000*np.log(31337 + i))) for i in range(100)]).dot(list(map(j, n - 10^4 + 100 - np.arange(100))))

if __name__ == "__main__":
    ...
    print("Stage 3:")
    print("========")
    key = hashlib.sha256(str(j(ITERS)).encode()).digest()
```

Calculating `j` function seems obvious, because it is completely linear, and there is no `n`-polynomials like Step 2.

But the problem is that the size is 10000 this time. 

<br>

It is not possible to construct a 10000 * 10000 matrix due to memory limits and time complexity.

But using the charpoly() I mentioned in Step 1, this can be done with 10000 degreed polynomial.

<br>

However, constructing a matrix, and converting it into a polynomial like Step 1. is not possible, we have to directly construct the polynomial.

It took some time for me to understand why does Hellman's `The Matrix Revolutions`' solution work.

And fascinated the modulus polynomial is exactly equal to coefficients in the recurrence equation.

```
Pow Done.
Mul Done.
4884838814356754393675352922066305300889643
```

We can see `j(ITERS)`'s value is 4884838814356754393675352922066305300889643, and the challenge is finished.

<br><br>

In total, my solution code took around 1.5 minutes for Step 1,

DLP was fast, but the final brute force part took the longest.

And less than 10 seconds for Step 2 and Step 3.

Here is my final solve code: [ex.sage](https://github.com/soon-haari/soon-haari.github.io/blob/master/files/functional/ex.sage)

<br>

The fact you don't even know if your ITERS and S3 is correct or not during solving makes the challenge a hundred times harder. 

I always love incredibly hard challenges with short source codes.

Thanks to this, I am planning to create a challenge of my own with some additional tweaks later.

Huge respect to Robin, the author again.

<br>

Flag: `ICC{N0w_y0u_4re_a_mast3r_0f_t3h_l1n34r_r3curr3nc3s!}`





