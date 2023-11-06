---
layout: post
category: studies
---

I co-authored a challenge with `ks` for Hspace CTF, which was Belluminar-type.

No team solved it during the CTF, but `Jinseo Kim` and `diff` from `GoN` solved it after the CTF ended.

### Description
```
FizzBuzz라는 게임을 아시나요? 감사하게도, Slyfizz께서 살신성인(殺身成仁)을 통해 이 챌린지의 마스코트가 되어 주셨습니다!

SlyFizzbuzz를 공부하여서 100번 연속 Slyfizz의 무빙을 맞춰보세요!

nc x.x.x.x 1013
```

### chall.py
```python
import random
import os

FLAG = open("flag", "r").read()

def fizzbuzz(n):
    fb = "sly"
    if n % 3 == 0:
        fb += "fizz"
    if n % 5 == 0:
        fb += "buzz"
    return fb

for rounds in range(100000):
    cmd = input("> ")
    if cmd == "roll":
        dice = random.getrandbits(8)
        print(fizzbuzz(dice))

    else:
        for _ in range(100):
            assert fizzbuzz(random.getrandbits(8)) == input("Guess> ")
        break
else:
    exit()

print(f"Here is your flag: {FLAG}")
```

The challenge extracts `random.getrandbits(8)`, and tell us if the number is multiple of 3 or multiple of 5. Which is not much different from the original [Fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz) game.

The goal is to recover the state, and predict `random.getrandbits(8)`'s slyfizzbuzz result 100 times.

If you are familiar with Python's random, which is implemented with the Mersenne Twister, you probably know that we can recover the twister's state if we can calculate 19937 bits from the output.

<br>

If any of the quotients was an even number, and the multiple flag is set to *True*, we can easily know that one LSB is equal to zero, because it is an even number. And easily recover one bit. Unfortunately, 3 and 5 is both an odd number.

<br>

The output `"slyfizzbuzz"` happens around in 1/15 probability, because the number has to be a multiple of 15.

Let's observe those numbers within `range(0, 256)`.

```python
for i in range(256):
    if i % 15 == 0:
        bin_str = bin(i)[2:]
        bin_str = "0" * (8 - len(bin_str)) + bin_str
        print(bin_str + f": {i}")
```
```
00000000: 0
00001111: 15
00011110: 30
00101101: 45
00111100: 60
01001011: 75
01011010: 90
01101001: 105
01111000: 120
10000111: 135
10010110: 150
10100101: 165
10110100: 180
11000011: 195
11010010: 210
11100001: 225
11110000: 240
11111111: 255
```

Except for 0 and 255, those numbers have an interesting common property.

Sum of upper 4 bits and lower 4 bits is always equal to `0b1111`.

<br>

This interesting property happens because 15 is exactly one less than 16.

By adding 15 from the previous number, it adds 1 to upper 4 bits, and subtracts 1 from lower 4 bits.

So until `0b11110000`, the sum of both 4 bits remains `0b1111`.

In the other form, so that we can extract information from the result, `bin(i)[:4] ^ bin(i)[4:] == 0b1111`.

(`^` means XOR here.)

<br>

But 0 and 255 doesn't satisfy that property. Instead, `bin(i)[:4] ^ bin(i)[4:] == 0b0000` for those 2 exceptional cases.

Fortunately, all 4 bits of `0b1111` and `0b0000` are equal, so we can extract 3 bits of information from that.

My implementation looks like:
```python
diff = Twister._xor(dat[:4], dat[4:])
diff = Twister._xor(diff[:3], diff[1:])

solver.insert(diff[0], 0)
solver.insert(diff[1], 0)
solver.insert(diff[2], 0)
```

There are 18 multiples of 15 in `range(0, 256)` so the probability is 18/256. And we can get 3 bits of information from that.

On average, given 100000 - 1 results, around 21093.5 bits of information is recieved, which is sufficient to recover the Mersenne Twister state.

<br>

If we only insert information bits of 0 instead of 1, the correct state, and the zero state(which all bits are 0) always coexist.

The correct one can be selected manually, but the easier(and much correcter, is that even a word?) way is to set the initial state before inserting bits.

Since every initial random state works with seed, twister's index is equal to 624, and first value of twister is equal to `0x80000000`. You don't believe me?

```
Python 3.11.1 (v3.11.1:a7a450f84a, Dec  6 2022, 15:24:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import random
>>> random.getstate()[1][0]
2147483648
>>>
```

So we set the initial state with the code below.

```python
twister = Twister()
solver = Solver()

twister.index = 624

solver.insert(twister.state[0][0], 1)
for i in range(1, 32):
    solver.insert(twister.state[0][i], 0)
```

<br>

This is my final solve code.

It should take less than 15 minutes on most laptops.

### ex.py
```python
import random
from pwn import *

class Twister:
    N = 624
    M = 397
    A = 0x9908b0df

    def __init__(self):
        self.state = [ [ (1 << (32 * i + (31 - j))) for j in range(32) ] for i in range(624)]
        self.index = 0
    
    @staticmethod
    def _xor(a, b):
        return [x ^ y for x, y in zip(a, b)]
    
    @staticmethod
    def _and(a, x):
        return [ v if (x >> (31 - i)) & 1 else 0 for i, v in enumerate(a) ]
    
    @staticmethod
    def _shiftr(a, x):
        return [0] * x + a[:-x]
    
    @staticmethod
    def _shiftl(a, x):
        return a[x:] + [0] * x

    def get32bits(self):
        if self.index >= self.N:
            for kk in range(self.N):
                y = self.state[kk][:1] + self.state[(kk + 1) % self.N][1:]
                z = [ y[-1] if (self.A >> (31 - i)) & 1 else 0 for i in range(32) ]
                self.state[kk] = self._xor(self.state[(kk + self.M) % self.N], self._shiftr(y, 1))
                self.state[kk] = self._xor(self.state[kk], z)
            self.index = 0

        y = self.state[self.index]
        y = self._xor(y, self._shiftr(y, 11))
        y = self._xor(y, self._and(self._shiftl(y, 7), 0x9d2c5680))
        y = self._xor(y, self._and(self._shiftl(y, 15), 0xefc60000))
        y = self._xor(y, self._shiftr(y, 18))
        self.index += 1

        return y
    
    def getrandbits(self, bit):
        return self.get32bits()[:bit]

    def randbytes(self, n):
        left = n

        res = []

        while left >= 4:
            left -= 4
            q = self.get32bits()

            for i in range(4):
                res.append(q[(3 - i) * 8:(4 - i) * 8][::-1])

        if left:
            q = self.get32bits()
            for i in range(left):
                res.append(q[(left - 1 - i) * 8:(left - i) * 8][::-1])

        assert len(res) == n

        return res

class Solver:
    def __init__(self):
        self.equations = []
        self.outputs = []
    
    def insert(self, equation, output):
        for eq, o in zip(self.equations, self.outputs):
            lsb = eq & -eq
            if equation & lsb:
                equation ^= eq
                output ^= o
        
        if equation == 0:
            if output == 0:
                return
            raise ValueError("Impossible generated bits.")

        lsb = equation & -equation
        for i in range(len(self.equations)):
            if self.equations[i] & lsb:
                self.equations[i] ^= equation
                self.outputs[i] ^= output
    
        self.equations.append(equation)
        self.outputs.append(output)
    
    def solve(self):
        num = 0
        for i, eq in enumerate(self.equations):
            if self.outputs[i]:
                # Assume every free variable is 0
                num |= eq & -eq

        
        state = [ (num >> (32 * i)) & 0xFFFFFFFF for i in range(624) ]
        return state

def fizzbuzz(n):
    fb = "sly"
    if n % 3 == 0:
        fb += "fizz"
    if n % 5 == 0:
        fb += "buzz"
    return fb

if __name__ == "__main__":

    io = remote("localhost", 1013)

    size = 5000

    twister = Twister()
    solver = Solver()

    twister.index = 624

    solver.insert(twister.state[0][0], 1)
    for i in range(1, 32):
        solver.insert(twister.state[0][i], 0)

    cnt = 0

    while True:
        print("Getting Data...")
        res = [0] * size

        for i in range(size):
            io.sendline(b"roll")
        for i in range(size):
            io.recvuntil(b"> ")
            if io.recvline()[:-1] == b"slyfizzbuzz":
                res[i] = 1

        cnt += size

        dats = [twister.getrandbits(8) for _ in range(size)]

        for i in range(size):
            dat = dats[i]
            if res[i] == 0:
                continue

            diff = Twister._xor(dat[:4], dat[4:])
            diff = Twister._xor(diff[:3], diff[1:])

            solver.insert(diff[0], 0)
            solver.insert(diff[1], 0)
            solver.insert(diff[2], 0)

            rank = len(solver.equations)

            print(f"{rank = }", end = "\r")

            if rank == 19968:
                break

        if rank == 19968:
            break

    state = solver.solve()

    random.setstate((3, tuple(state + [624]), None))

    for _ in range(cnt):
        random.getrandbits(8)

    io.sendlineafter(b"> ", b"asdf")

    for _ in range(100):
        io.sendlineafter(b"Guess> ", fizzbuzz(random.getrandbits(8)).encode())



    io.interactive()
```

### flag

`hspace{Fizzer_Buzzer_Chicken_Slyfizzbuzzer}`
