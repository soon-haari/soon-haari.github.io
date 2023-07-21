---
layout: post
category: studies
---

<br><br>

I didn't solve this challenge during the Quals, didn't even know 'D' of 'Differential Cryptanalysis' back then.

But I solved this without much trouble yesterday.

My solution was little bit different from usual DC though. So I share mine here.

<br>

### Description
- I made SusCipher, which is a vulnerable block cipher so everyone can break it! Please, try it and find a key.
- Hint: Differential cryptanalysis is useful.

### task.py

```python
#!/usr/bin/env python3
import hashlib
import os
import signal


class SusCipher:
    S = [
        43,  8, 57, 53, 48, 39, 15, 61,
         7, 44, 33,  9, 19, 41,  3, 14,
        42, 51,  6,  2, 49, 28, 55, 31,
         0,  4, 30,  1, 59, 50, 35, 47,
        25, 16, 37, 27, 10, 54, 26, 58,
        62, 13, 18, 22, 21, 24, 12, 20,
        29, 38, 23, 32, 60, 34,  5, 11,
        45, 63, 40, 46, 52, 36, 17, 56
    ]

    P = [
        21,  8, 23,  6,  7, 15,
        22, 13, 19, 16, 25, 28,
        31, 32, 34, 36,  3, 39,
        29, 26, 24,  1, 43, 35,
        45, 12, 47, 17, 14, 11,
        27, 37, 41, 38, 40, 20,
         2,  0,  5,  4, 42, 18,
        44, 30, 46, 33,  9, 10
    ]

    ROUND = 3
    BLOCK_NUM = 8
    MASK = (1 << (6 * BLOCK_NUM)) - 1

    @classmethod
    def _divide(cls, v: int) -> list[int]:
        l: list[int] = []
        for _ in range(cls.BLOCK_NUM):
            l.append(v & 0b111111)
            v >>= 6
        return l[::-1]

    @staticmethod
    def _combine(block: list[int]) -> int:
        res = 0
        for v in block:
            res <<= 6
            res |= v
        return res

    @classmethod
    def _sub(cls, block: list[int]) -> list[int]:
        return [cls.S[v] for v in block]

    @classmethod
    def _perm(cls, block: list[int]) -> list[int]:
        bits = ""
        for b in block:
            bits += f"{b:06b}"

        buf = ["_" for _ in range(6 * cls.BLOCK_NUM)]
        for i in range(6 * cls.BLOCK_NUM):
            buf[cls.P[i]] = bits[i]

        permd = "".join(buf)
        return [int(permd[i : i + 6], 2) for i in range(0, 6 * cls.BLOCK_NUM, 6)]

    @staticmethod
    def _xor(a: list[int], b: list[int]) -> list[int]:
        return [x ^ y for x, y in zip(a, b)]

    def __init__(self, key: int):
        assert 0 <= key <= self.MASK

        keys = [key]
        for _ in range(self.ROUND):
            v = hashlib.sha256(str(keys[-1]).encode()).digest()
            v = int.from_bytes(v, "big") & self.MASK
            keys.append(v)

        self.subkeys = [self._divide(k) for k in keys]

    def encrypt(self, inp: int) -> int:
        block = self._divide(inp)

        block = self._xor(block, self.subkeys[0])
        for r in range(self.ROUND):
            block = self._sub(block)
            block = self._perm(block)
            block = self._xor(block, self.subkeys[r + 1])

        return self._combine(block)

    # TODO: Implement decryption
    def decrypt(self, inp: int) -> int:
        raise NotImplementedError()


def handler(_signum, _frame):
    print("Time out!")
    exit(0)


def main():
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(300)
    key = int.from_bytes(os.urandom(6), "big")

    cipher = SusCipher(key)

    while True:
        inp = input("> ")

        try:
            l = [int(v.strip()) for v in inp.split(",")]
        except ValueError:
            print("Wrong input!")
            exit(0)

        if len(l) > 0x100:
            print("Long input!")
            exit(0)

        if len(l) == 1 and l[0] == key:
            with open('flag', 'r') as f:
                print(f.read())

        print(", ".join(str(cipher.encrypt(v)) for v in l))


if __name__ == "__main__":
    main()
```

The goal is to find the key.

<br><br>

As far as I know, the usual process of DC is to find last subkeys with statistics, and second last... and so on.

I applied a way to directly get the first subkey, which is our goal.

<br><br>

## idea

After observing the properties of S-box, I tried sending two plaintexts with just one bit differential.

For example, I set input differential to `[1, 0, 0, 0, 0, 0, 0, 0]`

<br>

And the output differential came out as following:

```
...
[34, 53, 32, 44, 12, 33, 11, 8]
[0, 0, 0, 8, 45, 27, 55, 14]
[0, 1, 1, 8, 0, 26, 9, 4]
[45, 5, 13, 32, 4, 34, 55, 13]
[16, 0, 41, 0, 0, 1, 41, 20]
[0, 0, 1, 0, 40, 8, 0, 5]
[0, 1, 32, 8, 45, 26, 51, 36]
[27, 24, 33, 45, 41, 36, 0, 42]
[26, 3, 33, 8, 1, 1, 0, 37]
[7, 39, 33, 37, 37, 59, 20, 39]
[11, 0, 9, 0, 5, 1, 2, 33]
[32, 33, 0, 4, 13, 63, 46, 37]
[29, 4, 1, 40, 12, 32, 63, 28]
[0, 56, 0, 4, 0, 32, 27, 10]
[0, 0, 4, 0, 0, 0, 0, 0]
...
```

Nothing much special, but the point I focused on is that

There are too many `0`s compared to the amount they should exist as an ideal cipher.

<br>

The reason for this is, obviously, there are too few rounds(3).

Also, I calculated the average rate of how many nonzeros exist in the output differential.

<br><br>

With input differential `[1, 0, 0, 0, 0, 0, 0, 0]`,

Approximately `6.04727` nonzeros existed in the output differential. (100000 samples)

<br>

With input differential `[63, 0, 0, 0, 0, 0, 0, 0]`,

Approximately `6.34538` nonzeros existed in the output differential. (100000 samples)

<br><br>

We can see that there is not 'big', but meaningful difference between two values.

<br><br>

## Estimating the number of nonzeros


Now let's advance this idea.

The step of encryption is

<center>
xor0 -> sub0 -> perm0 ->
<br>
xor1 -> sub1 -> perm1 ->
<br>
xor2 -> sub2 -> perm2 ->
<br>
xor3
</center>

Let's say we input two plaintexts `a`, `b`.

<br>

If we assume that we know the value of `key0[0]` (0 ~ 63), 

We know the exact value of `a[0]` and `b[0]` right after `xor0`.

(During normal DC, we only know `a[0] ^ b[0]`)

<br>

Then we can calculate `S[a[0]]` and `S[b[0]]`.

Which means we can know the value of `a[0] ^ b[0]` right after `sub0`.

Other 7 values of differential of `a` and `b` will still remain with 0.

<br>

My idea here, is to try all those possible differentials, and find the average number of nonzeros of the final differential.

There are only 64 possibilities of the initial differential  which is `[k, 0, 0, 0, 0, 0, 0, 0]`, `k = 0 ~ 63`, so calculating all of them is fair enough.

(`k = 0` is actually impossible but doesn't matter.)

<br>

The position we are on is right after `sub0`, so still 2 substitution is left.

How can we reduce it to 1?

<br><br>

From the result of encryption of `a`, `b`, we can calculate the differential of `enc(a)` and `enc(b)`.

Let's think reversed now.

- `xor3` doesn't change anything to the differential. 

- `perm2` is operated same to both texts, so just perm_inverting once will give us the differential right after `sub2` is finished.

- `sub2` is the part we can't be sure of anything, but there is this import property of substitution: Nonzero differential is kept nonzero, and zero differential is kept zero.

Which means number of nonzero values of differential right before `sub2` is exactly equal to right after `sub2`.

(That's why I focused on number of nonzeros.)

<br><br>

So we can conclude that we only have to process until `xor2` which is right before `sub2`.

Again, adding key doesn't affect the differential, so until `perm1` should be enough.

That gives us:
<center>
perm0 -> xor1 -> sub1 -> perm1
</center>
to do.

We can do nothing about the xor1 part, so I decided to just ignore.

```python
weight = [0] * 64

for i in trange(sample_num):
    a = [random.randrange(0, 64) for _ in range(8)]
    for k in range(64):
        b = a[:]
        b[idx] ^= k

        a_ = a[:]
        b_ = b[:]

        a_ = SusCipher._perm(a_)
        b_ = SusCipher._perm(b_)

        a_ = SusCipher._sub(a_)
        b_ = SusCipher._sub(b_)

        a_ = SusCipher._perm(a_)
        b_ = SusCipher._perm(b_)

        weight[k] += not0count(SusCipher._xor(a_, b_))

for i in range(64):
    weight[i] /= sample_num

print(weight)
```

Function `not0count` is defined as:
```python
def not0count(a):
    res = 0
    for k in a:
        if k:
            res += 1

    return res
```

<br><br>

This is the result of `weight` I calculated with `sample_num = 1000`

```
[0.0, 2.403, 2.227, 4.63, 2.572, 4.975, 2.539, 4.942, 2.291, 3.937, 3.926, 5.572, 4.059, 5.705, 4.0, 5.646, 2.326, 4.729, 2.347, 4.75, 2.256, 4.659, 2.172, 4.575, 4.056, 5.702, 3.904, 5.55, 3.877, 5.523, 3.773, 5.419, 2.638, 4.156, 4.284, 5.802, 4.416, 5.934, 4.36, 5.878, 2.417, 4.13, 4.049, 5.762, 4.188, 5.901, 4.131, 5.844, 4.399, 5.917, 4.263, 5.781, 4.234, 5.752, 4.14, 5.658, 4.175, 5.888, 4.027, 5.74, 4.01, 5.723, 3.913, 5.626]
```

The result is very very surprising because it varies a lot from value to value.

Result of diff = 0 remains 0 obviously.

Result of diff = 1, 2, 4, 8, 16, 32 are all 2.xxx(which is very small compared to others.), because they consists on only 1 bit.

diff of many bits, like 63's result is very high, like 5.626.

<br>

With more samples, the numbers would work better.

<br><br>

## Finding the key values

Then how can we use these weights to get the key?

Let's say the real `key0[0]` is equal to `k`.

And set `a` as `[a_, p1, p2, p3, p4, p5, p6, p7]`,

Set `b` as `[b_, p1, p2, p3, p4, p5, p6, p7]`

<br>

As I mentioned before, the differential after `sub0` would be

`[S[a_ ^ k] ^ S[b_ ^ k], 0, 0, 0, 0, 0, 0, 0]`.

We iterate all 64 possibilities of `k`, and check if the result's number of nonzero is similar to the average result we generated on `weight`.

If the `k` is correct, the difference should be smaller than others.

<br>

I made another table called `key_chance`, and added the difference between estimation from `weight`, and real result from the output of `task.py`.

```python
key_chance = [0] * 64
for i in range(send_n):
    a, b = st[i]
    final = not0count(SusCipher._perm_inv(SusCipher._divide(res1[i] ^ res2[i])))

    for k in range(64):
        init_dif = SusCipher.S[a ^ k] ^ SusCipher.S[b ^ k]

        key_chance[k] += abs(weight[init_dif] - final)
```
Note: `SusCipher._perm_inv` isn't implemented, so I had to implement it myself with inverse PBox.


`key_chance`'s result looks like:

```
[29404.817000000727, 30679.350000000642, ... 30124.760000000537, 14848.692999999996, 29970.971000000598, ... 30145.086000000614, 30154.163000000608]
```

Most of them has similar results, but exactly one has the value of half compared to others.

Which is a good news.

<br>

Repeat this 8 times for all `idx`, and we can get the key.

```
[*] Switching to interactive mode
soon_haari{I'm_such_a_good_surfer_ha_ha_ha}
28491066801130
> $
[*] Interrupted
```

Sadly, the server is closed, I can't get the real flag anymore.

Here is my final exploit code: [ex.py](https://github.com/soon-haari/soon-haari.github.io/blob/master/files/suscipher/ex.py)

<br><br><br>

## Finally..

At first, I didn't actually think this would actually work.

But again, thought why it wouldn't?

<br>

I think this is appliable for all kind of low-round ciphers, with lots of `0`s in output differential.

If I solved this one during the Quals, I would have scored `990`, which is a nice rank in Korea.

Sad, but also somehow proves that I developed a lot compared to back then.

<br>

Huge respect to RBTree, the author.











