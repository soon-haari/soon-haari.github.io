---
layout: post
category: studies
---

<br><br>

CTF에서는 난수들의 정보를 몇 개 얻어서 다음 난수들의 정보를 알아내는 문제가 종종 출제되곤 합니다.

Cryptography 분야인 만큼, 주로 Python으로 구현된 문제들이고, 구현되어 있는 random 모듈을 사용합니다.

언어별로 사용하는 랜덤의 구현 방식 또한 각기 다르기 때문에 오늘은 Python의 random에 대해서 다루어보도록 하겠습니다.

뚫는 방식이 비교적 비슷한 js의 random, xorshift128까지 추가적으로 다루어보겠습니다.

<br>

영어권보다도 한국 독자들의 시점에 맞춰 처음으로 한글로 작성해보겠습니다.

최대한 이해가 쉽게 차근차근 설명해보는 것을 목표로 하고 있습니다.

\* 중간중간에 수식과 코드가 있어 모바일 환경에서 읽는 것은 추천드리지 않습니다. \*

<br><br>

## **Python's random**

#### **1\. Python's random module**

Python의 random은 메르센 트위스터로 구현되어 있다는 것은 많은 분들이 아시리라 생각됩니다. 하지만 백문이 불여일견, 직접 뜯어서 확인해보는 것만큼 좋은 공부는 없다고 생각됩니다.

간단한 구글링을 통해서도 ([https://github.com/python/cpython/blob/main/Lib/random.py](https://github.com/python/cpython/blob/main/Lib/random.py)) 구현을 쉽게 찾을 수 있습니다.

또는 저처럼 인터넷을 믿지 않는 사람은 직접 random.py를 열어보는 것도 좋은 생각입니다.

```
Python 3.10.7 (v3.10.7:6cc6b13308, Sep  5 2022, 14:02:52) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import random
>>> random
<module 'random' from '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/random.py'>
```

```python
class Random(_random.Random):
    """Random number generator base class used by bound module functions.

    Used to instantiate instances of Random to get generators that don't
    share state.

    Class Random can also be subclassed if you want to use a different basic
    generator of your own devising: in that case, override the following
    methods:  random(), seed(), getstate(), and setstate().
    Optionally, implement a getrandbits() method so that randrange()
    can cover arbitrarily large ranges.

    """
```

하지만 이 모듈은 사실상 random.getrandbits 함수를 기반으로 어떤 방식으로

-   범위내의 난수를 생성하고 (random.randrange)
-   0~1의 실수를 생성하고 (random.random)
-   배열을 섞는지 (random.shuffle)

그리고 random.choice, random.sample 등의 다양한 유용한 함수들.. 의 구현을 관찰할 뿐, 직접적으로 메르센 트위스터의 내부 구현은 \_random에 구현되어 있습니다.

저희에게 중요한 random.seed, random.getstate, random.setstate와 같은 함수도 super()을 사용해 더 이상 알아낼 수 있는 것이 없습니다.

```python
	def seed(self, a=None, version=2):
        """Initialize internal state from a seed.
		...생략...
        """
        super().seed(a)
        self.gauss_next = None
```

\_random부터는 Python이 아닌 라이브러리의 구현을 사용하기 때문에 검색을 하는 것이 바람직합니다.

```
>>> import _random
>>> _random
<module '_random' from '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload/_random.cpython-310-darwin.so'>
```

위에서 언급한 3개의 함수는 뒤에서 CTF 기출과 함께 다시 다루도록 하겠습니다.

<br>

#### **2\. Mersenne Twister**

[https://github.com/eboda/mersenne-twister-recover/blob/master/MersenneTwister.py](https://github.com/eboda/mersenne-twister-recover/blob/master/MersenneTwister.py)

위 링크에서 Python random의 완전 파이썬 구현을 찾을 수 있었습니다. 친절하게 test\_PythonMT19937 함수를 통해

여기에 구현되어 있는 MT19937.extract 함수와 getrandbits(32)가 동일한 결과를 내고, PythonMT19937.seed가 올바르게 작동하고 있다는 것 또한 알려줍니다.

MT19937 class는 클래식한 Mersenne Twister의 구현이고 [https://en.wikipedia.org/wiki/Mersenne\_Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)에 있는 Pseudo Code와 동일합니다.

PythonMT19937은 거기에 올바른 random.seed 기능만을 추가로 구현해준 상속 class입니다.

<br>

그러면 먼저 MT19937 class를 분석해봅시다.

```python
class MT19937:
    """Classical Mersenne Twister Implementation."""

    def __init__(self, seed=None):
        self.mt = [0 for i in range(624)]
        self.index = 624
        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        self.mt[0] = seed
        for i in range(1, 624):
            self.mt[i] = self._int32(0x6c078965 *
                                (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i)

    def extract(self):
        """ Extracts a 32bit word """
        if self.index >= 624:
            self.twist()

        x = self.mt[self.index]
        x ^= x >> 11
        x ^= (x << 7) & 0x9d2c5680
        x ^= (x << 15) & 0xefc60000
        x ^= x >> 18

        self.index += 1
        return self._int32(x)

    def twist(self):
        """ The twist operation. Advances the internal state """
        for i in range(624):
            upper = 0x80000000
            lower = 0x7fffffff

            x = self._int32((self.mt[i] & upper) +
                            (self.mt[(i + 1) % 624] & lower))
            self.mt[i] = self.mt[(i + 397) % 624] ^ (x >> 1)

            if x & 1 != 0:
                self.mt[i] ^= 0x9908b0df

        self.index = 0

    def _int32(self, x):
        return x & 0xffffffff
```

대략적인 과정을 요약해보면

-   32비트 값 624개를 기준으로 항상 state가 존재, 즉 19968비트
-   extract 과정에서는 값 한개만을 뽑아서 shift, and, xor이런 연산들을 진행하고 return
-   extract는 624개의 값들을 앞에서부터 하나씩 진행
-   624회가 모두 끝나면 twist를 통해 624개 값 전부 업데이트

가 됩니다.

그럼 extract과정과 twist과정 분석을 진행해야 하는데, 먼저 extract를 한 번 봐봅시다.

```python
	def extract(self):
        """ Extracts a 32bit word """
        if self.index >= 624:
            self.twist()

        x = self.mt[self.index]
        x ^= x >> 11
        x ^= (x << 7) & 0x9d2c5680
        x ^= (x << 15) & 0xefc60000
        x ^= x >> 18

        self.index += 1
        return self._int32(x)
```

- x를 오른쪽으로 11비트 쉬프트한걸 xor,

- 왼쪽으로 7비트 쉬프트하고 어떤 mask에 넣은걸 xor,

- 왼쪽으로 15비트 쉬프트하고 mask에 넣은걸 xor,

- 오른쪽으로 18비트 쉬프트하고 xor

한 결과를 최종적으로 출력해줍니다.

<br>

이 연산들이 모두 비트연산이라는 사실은 우리에게 random을 뚫을 수 있는 큰 도움이 됩니다.

지금 초기에 있는 수가 32비트 정수인데, 이 정수를 한 값이 아닌, 1비트 변수 32개로 생각해봅니다.

이 extract 과정을 tamper이라고 하는데, tamper 과정이 끝난 후에도 32비트 정수가 배출되고, 그걸 리턴함으로서 비로소 우리에게 getrandbits(32)가 들어오게 됩니다.

그러면 tamper 후의 값도 1비트 변수 32개로 표현할 수 있을텐데, tamper 전후의 32개의 변수 사이에는 어떤 관계가 있을까요?

<br>

간단히 위 코드 상의 초기 x를 $x\_{0}, x\_{1}, \\ldots ,x\_{31}$이라고 놓습니다.

그렇다면 x >> 11은 다음과 같이 표현될 것입니다.

<center>$0, 0,\ldots,0, x_{0}, x_{1}, \ldots x_{20}$</center>

그렇다면 x ^= x >> 11을 실행한 후의 x의 모습은

<center>$x_{0}, x_{1},\ldots, x_{10}, x_{11} + x_{0}, x_{12} + x_{1}, \ldots , x_{31} + x_{20}$</center>

과 같은 모습이 됩니다.

<br>

여기서 xor연산을 덧셈으로 표현할 수 있는 이유는 각 비트를 order 2의 Galois Field와 같은 방식으로 생각하였기 때문입니다.

xor보다도 이렇게 덧셈으로 표현하면 편리한 이유가 등장합니다.

이 tamper 과정 중 등장하는 연산은 xor, left shift, right shift, and 연산밖에 존재하지 않습니다.

이 연산들은, 아무리 여러번 순서를 섞어 실행해도 변수들의 선형성을 유지한다는 특징을 가지고 있습니다.

즉 아무리 복잡한 tamper 과정을 거쳐도 tamper 후의 1비트 변수 32개는 각각

초기에 설정한$x\_{0}, x\_{1}, \\ldots ,x\_{31}$에 대해

<center>$a_{0}x_{0} + a_{1}x_{1} + \ldots + a_{31}x_{31}$</center>
꼴로 표현할 수가 있다는 뜻입니다.

심지어 우리는 GF(2)위에서 연산을 수행중이기 때문에 모든 $a\_{0}, a\_{1}, \\ldots ,a\_{31}$은 0 또는 1로만 표현될 수 있어서 더욱 편리해집니다.

이를 간단히 $tamper\_{0}, tamper\_{1}, \\ldots ,tamper\_{31}$로 쓰도록 하겠습니다.

<br><br>

여기서 중요한 한 발짝을 더 나아가야 하는데, 이 부분부터는 선형대수학을 공부하셨다면 더 쉽게 이해하실 수 있을 것으로 생각됩니다.

위에서 언급한 선형합으로 나타낼 수 있는 식은

<center>$tamper_{i} = \begin{bmatrix}a_{0} & a_{1} & \ldots & a_{30} & a_{31}\end{bmatrix}\begin{bmatrix}x_{0}\\ x_{1}\\ \vdots\\ x_{30}\\ x_{31}\end{bmatrix}$</center>

로 표현 가능합니다.

그런데 이 $a\_{0}, a\_{1}, \\ldots ,a\_{31}$이라는 계수들은 tamper 후의 32개의 변수별로 각각 다를 것입니다.

즉 최종적으로 쓸 수 있는 식은

<center>$\begin{bmatrix}tamper_{0}\\ tamper_{1}\\ \vdots\\ tamper_{30}\\ tamper_{31}\end{bmatrix}=\begin{bmatrix}a_{0, 0} & a_{0, 1} & \ldots & a_{0, 30} & a_{0, 31}\\ a_{1, 0} & a_{1, 1} & \ldots & a_{1, 30} & a_{1, 31}\\ \vdots & \vdots & \ddots & \vdots & \vdots\\ a_{30, 0} & a_{30, 1} & \ldots & a_{30, 30} & a_{30, 31}\\ a_{31, 0} & a_{31, 1} & \ldots & a_{31, 30} & a_{31, 31}\end{bmatrix}\begin{bmatrix}x_{0}\\ x_{1}\\ \vdots\\ x_{30}\\ x_{31}\end{bmatrix}$</center>

이라는 행렬식을 쓸 수 있습니다. 32 \* 32개의 a값들은 정해진 tamper과정을 통해서 기존 x값과 상관없이 구할 수 있습니다.

일반적으로 이 tamper 과정은 1대1 대응이기 때문이기 때문에 untamper이 가능하다고 알려져 있고, untamper을 진행하는 함수 또한 존재합니다. 역연산이 가능하다는 것과 위에서 만든 32 \* 32 사이즈의 행렬의 역행렬이 존재하는 것과 동치라는 것을 유추할 수 있습니다.

<br>

하지만 624개의 연속한 full 32비트 값들을 알지 않는 이상, untamper만으로 state를 복구하는 것은 쉽지만은 않습니다.

실제 CTF에 등장하는 문제들에서 알 수 있는 값들로는 32비트 중에서 몇 개의 비트밖에는 알아낼 수 없는 경우가 대부분입니다.

즉 untamper하는 과정을 아는 것보다는 tamper이라는 과정이 그냥 기존 32개의 비트들을 선형결합하는 과정이다 정도만 이 부분에서 이해하셨으면 좋겠습니다. 이 개념 자체가 이제 등장할 19968변수 연립방정식의 모든 것이기 때문입니다. 

사실 만약 untamper이 불가능하게 tamper이 구현되어 있더라도(즉, 비트손실이 일어나게), 결과적으로 state를 복구하는 데는 큰 문제가 없을 것으로 생각됩니다.

<br>

이제 twist 함수를 관찰해봅시다.

```python
    def twist(self):
        """ The twist operation. Advances the internal state """
        for i in range(624):
            upper = 0x80000000
            lower = 0x7fffffff

            x = self._int32((self.mt[i] & upper) +
                            (self.mt[(i + 1) % 624] & lower))
            self.mt[i] = self.mt[(i + 397) % 624] ^ (x >> 1)

            if x & 1 != 0:
                self.mt[i] ^= 0x9908b0df

        self.index = 0
```

이 또한 위에서 등장한 '안전한' 비트 연산들인 xor, shift, and 연산만이 등장하는 것을 볼 수 있습니다. extract, twist 과정 모두 그 과정을 알아야 하는 것이 아니라 선형성이 보장되는 연산이라는 것만 이해하면 됩니다.

그런데 아까는 한 32비트 정수, 즉 1비트 변수 32개만을 가지고 업데이트를 하는 반면, 이번에는 624개의 값이 전부 등장해서 복잡한 업데이트를 진행합니다.

조금 사이즈가 커지긴 했지만 이 또한 당황하지 않고 아까와 같이 생각하면 됩니다. 

32 \* 624 = 19968개의 변수로 바뀐 것일 뿐입니다.

<center>$\begin{bmatrix}mt'_{0}\\ mt'_{1}\\ \vdots\\ mt'_{19966}\\ mt'_{19967}\end{bmatrix}=\begin{bmatrix}a_{0, 0} & a_{0, 1} & \ldots & a_{0, 19966} & a_{0, 19967}\\ a_{1, 0} & a_{1, 1} & \ldots & a_{1, 19966} & a_{1, 19967}\\ \vdots & \vdots & \ddots & \vdots & \vdots\\ a_{19966, 0} & a_{19966, 1} & \ldots & a_{19966, 19966} & a_{19966, 19967}\\ a_{19967, 0} & a_{19967, 1} & \ldots & a_{19967, 19966} & a_{19967, 19967}\end{bmatrix}\begin{bmatrix}mt_{0}\\ mt_{1}\\ \vdots\\ mt_{19966}\\ mt_{19967}\end{bmatrix}$</center>

<br><br>

그러면 이러한 선형성을 이용해서 어떻게 mersenne twister의 미래 값을 예측할 수 있을까요?

미래 값을 예측한다는 것은 사실상 random의 초기 state(32비트 \* 624)를 안다는 것과 동치입니다. 초기 state를 안다면, 호출한 random 함수들의 값들이 모두 일치하는지 assertion 확인을 한 후, 그 뒤로 호출될 함수들을 그대로 예측할 수 있기 때문입니다.

그리고 MT19937 class를 다시 한 번 봐보면, self.\_ 꼴의 변수는 mt, index뿐인 것을 확인할 수 있습니다.

mt는 우리가 계속 가지고 놀고 있는 32비트 정수 624개이고, index는 현재의 index에 해당하는 값을 extract하면서, 624에 도달하면 twist를 실행해 mt를 업데이트하고 다시 앞에서부터 extract를 진행해주는, 현재 위치를 가리켜주는 변수입니다.

random.getstate와 random.setstate는 random.seed에 비해 굉장히 직관적입니다.

```python
(VERSION = 3, [mt(624개), index], {additional function})
```

위 형태의 tuple 꼴로 mt를 입출력받고, 대부분의 상황에서 parameter 1은 3, parameter 3은 None이 되어

32 \* 624 비트의 mt와 index만 조작해주면 되는 상황이 대부분입니다.

그래서 초기에 생성된 mt값들만 알더라도 바로 setstate를 통해서 우리가 원하는 random 함수들을 모두 사용할 수 있습니다.

<br><br>

그러면 초기 mt를 알아내는 방법을 고안해봅시다.

먼저 seed를 호출하고, extract를 아직 한 번도 호출하지 않은 상태, 그리고 index = 624인 상태에서의 mt의 변수들을 $x\_{0},x\_{1},\\ldots,x\_{19966},x\_{19967}$로 놓겠습니다. 모두 1비트로 0, 1의 값만을 가집니다.

그러면 이 상황에서 getrandbits(32)를 호출했다고 생각해보면, 그 결과의 각 비트들은 위에서 이해한 것과 같이 $a\_{0}x\_{0} + a\_{1}x\_{1} + \\ldots + a\_{19967}x\_{19967}$로 표현 가능할 것입니다.

여기서 $x\_{0},x\_{1},\\ldots,x\_{19966},x\_{19967}$만이 unknown 변수이고, $a\_{0},a\_{1},\\ldots,a\_{19966},a\_{19967}$는 저희가 아무 정보 없이 내부적으로 계산할 수 있는 상수임에 유의해야 합니다.

그렇다면 저희는 getrandbits의 임의의 한 비트만을 알아도 벌써 $x\_{0},x\_{1},\\ldots,x\_{19966},x\_{19967}$에 관한 하나의 방정식을 세울 수 있습니다.

<center>$\begin{bmatrix}r_{0}\end{bmatrix}=\begin{bmatrix}a_{0} & a_{1} & \ldots & a_{19966} & a_{19967}\end{bmatrix}\begin{bmatrix}x_{0}\\ x_{1}\\ \vdots\\ x_{19966}\\ x_{19967}\end{bmatrix}$</center>

그런데 만약 k개의 비트를 알 수 있다면 어떨까요, 당연히 한 getrandbits(32)에 있을 필요 없이 아무 상황에서나 32개중 몇 번째에 있는 아무 비트만 알아도 방정식이 추가됩니다.

이 정보들을 $r\_{0},r\_{1},\\ldots,r\_{k - 1}$로 놓으면 행렬방정식이 세워집니다.

<center>$\begin{bmatrix}r_{0}\\ \vdots\\r_{k-1}\end{bmatrix}=\begin{bmatrix}a_{0, 0} & a_{0, 1} & \ldots & a_{0, 19966} & a_{0, 19967}\\ \vdots & \vdots & \ddots & \vdots & \vdots\\a_{k-1, 0} & a_{k-1, 1} & \ldots & a_{k-1, 19966} & a_{k-1, 19967}\end{bmatrix}\begin{bmatrix}x_{0}\\ x_{1}\\ \vdots\\ x_{19966}\\ x_{19967}\end{bmatrix}$</center>

등장하는 모든 a값들은 메르센 트위스터의 작동 원리를 이용해서 구해주어야 합니다. 어려운 과정은 아니지만 성가신 작업인데, rbtree님께서 a값들을 쉽게 구할 수 있는 깜짝 놀랄만한 방법을 고안하셨습니다. 조금 뒤에 설명하도록 하겠습니다.

선형대수학과 sage에 익숙하신 분이라면 이제 저 식들을 푸는 것은 문제도 아니라는 것을 느끼셨을 듯 합니다. solve\_right과 같은 함수를 사용해 최고의 시간복잡도로 $x\_{0},x\_{1},\\ldots,x\_{19966},x\_{19967}$의 해를 구할 수 있습니다.

하지만 $GF(2)$위에서 행렬식을 만들어 solve\_right를 이용하는 것은 상대적으로 너무 많은 메모리를 사용하고, 소요 시간 또한 너무 비효율적이기에, rbtree님의 방정식 풀이는 충분히 소개할 가치가 있고 굉장히 유용하다고 판단하였습니다. 거기다가 z3을 이용한 풀이까지 코드를 짜서 소요 시간까지 한번 비교해보려고 합니다.

<br><br>

Mersenne twister에 대해 설명된 한글로 된 글은 [https://rbtree.blog/posts/2021-05-18-breaking-python-random-module/](https://rbtree.blog/posts/2021-05-18-breaking-python-random-module/) 이 글만큼 명료한 글이 없다고 생각됩니다. (이미 모두가 아시리라 생각됩니다.)

그 글에 등장하는 Mersenne Twister 솔브코드를 가지고 왔습니다.

```python
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
            return

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
```

Twister 클래스는 아까 언급한 a값들을 간편하게 구하는 클래스입니다. 내부 구현을 들여다보면 \_\_init\_\_ 함수 빼고는 그냥 메르센 트위스터 작동 방식과 동일합니다. 32비트를 정수가 아닌 값 32개로 된 배열로 표현했다는 점도 다릅니다.

그러면 이 Twister 클래스의 작동법을 살펴봅시다.

<br>

\_\_init\_\_에서 초기 state를 구성하는 19968개의 값들을 기존처럼 0, 1이 아닌,$2^{0}, 2^{1}, \\ldots, 2^{19966}, 2^{19967}$ 로 집어넣었습니다. 즉, xor의 계산을 계속 한다는 가정하에, 19968개의 비트들이 작은 비트부터 각자 하나씩 할당되어 있다고 할 수 있습니다.

그래서 정수값끼리 xor하는 과정을 방정식의 덧셈으로 생각할 수가 있습니다. xor은 속도가 빠른 연산이기에 속도 또한 polynomial이나 배열 등을 만들어 구현하는 것보다 훨씬 빠를 것으로 생각됩니다.

결과적으로 Twister 클래스의 get32bits 함수는 getrandbits(32)에 해당하는 32개의 비트들을 초기 19968개의 값의 선형적인 합의 계수를 비트로 표현한 19968비트 정수 32개를 return해줍니다.

<br>

Solver 클래스는 위에서 언급한, k개의 방정식이 등장하는 행렬방정식을 RREF(Row Reduced Echelon Form)으로 변환해주는 과정을 방정식을 하나씩 추가할 때마다 업데이트해주는 클래스입니다.

최종적으로 모든 방정식이 다 입력되었을 때는 leading 1이 존재하지 않는 column의 Free Variable을 0으로 설정한 초기state로 성립할 수 있는 한 가지의 해를 solve함수가 return해줍니다.

<br>
<center>-</center>
<br>

solve\_right, z3에 비해서 확실히 좋은 장점은, Free Variable의 위치를 직접적으로 알 수 있어서 어디의 비트가 상관없는지와 같은 세부적인 정보들을 알 수 있습니다. 즉, 하나의 해 뿐만 아니라 존재하는 모든 해들을 구할 수 있다고까지 말할 수 있습니다. (sage로도 right\_kernel을 사용하면 가능은 할 것이라고 생각됩니다.)

그리고 getrandbits(32)의 32비트중 특정 개수의 MSB만의 정보를 알 수 있을 때, 19968개의 비트가 항상 등장하지 않는다는 사실이 흥미로웠고, random.random()과 같이 26, 27개의 MSB만을 필요로 하는 함수를 이용한 문제에 큰 도움이 됩니다.

이는 '4. random의 함수들' 절에서 다시 언급하겠습니다.

이 부분은 rbtree님의 블로그를 참고하시면 좋을 듯 합니다.

<br>

직접 RREF, z3 중 어느 방법이 더 속도가 빠른지 확인하기 위해서 2가지 방법으로 state를 복구하는 함수를 작성해보았습니다.

rbtree님의 Solver 클래스에서 equation이 0일때 return하는 부분을 다음과 같이 조금 수정하였습니다.

```python
        if equation == 0:
            if output == 0:
                return
            raise ValueError("Impossible generated bits.")
```

equation이 0일 때 output이 1이라면 해가 존재하지 않기 때문에 계속 정보를 넣고 solve과정까지 가기 전에 집어넣은 비트에 오류가 있음을 알 수 있게 하기 위해서 위와 같은 수정을 하였습니다.

먼저 getrandbits(26)을 이용한 복구를 진행해보았습니다. 26개의 추후 MSB들을 알기 위해서는 1247개의 값이 필요하고, 따라서 twist는 한번만 진행해주어도 됩니다.

z3은 특정 상황에서 굉장히 빠르지만, twist와 같이 복잡한 연산들이 중첩되었을 때는 굉장히 느려질 수 있습니다.

```python
from z3 import *
import rbtree_mersenne
import random
import time

def rbtree_solve(outputs):
    twister = rbtree_mersenne.Twister()
    solver = rbtree_mersenne.Solver()

    for tup in outputs:
        res, bits = tup
        equation = twister.getrandbits(bits)

        for i in range(bits):
            solver.insert(equation[i], (res >> (bits - 1 - i)) & 1)

    state = solver.solve()
    return (3, tuple(state + [0]), None)



def z3_solve(outputs):
    MT = [BitVec(f'm{i}', 32) for i in range(624)]
    s = Solver()
    
    def tamper(y):
        y ^= LShR(y, 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ LShR(y, 18)
        
    def getnext():
        x = Concat(Extract(31, 31, MT[0]), Extract(30, 0, MT[1]))
        y = If(x & 1 == 0, BitVecVal(0, 32), 0x9908B0DF)
        MT.append(MT[397] ^ LShR(x, 1) ^ y)
        return tamper(MT.pop(0))
        
    def getrandbits(n):
        return Extract(31, 32 - n, getnext())
        
    s.add([getrandbits(op[1]) == op[0] for op in outputs])
    assert(s.check() == sat)
    state = [s.model()[x].as_long() for x in [BitVec(f'm{i}', 32) for i in range(624)]]
    
    return (3, tuple(state + [0]), None)


if __name__ == "__main__":

    bits = 26
    number = 1248
    # change these values for different tests

    output = []
    for _ in range(number):
        output.append((random.getrandbits(bits), bits))
    check = []
    for _ in range(10000):
        check.append(random.getrandbits(bits))


    st = time.time()
    random.setstate(rbtree_solve(output))
    for _ in range(number):
        assert output[_][0] == random.getrandbits(bits)
    for _ in range(10000):
        assert check[_] == random.getrandbits(bits)
    fin = time.time()
    print(f"rbtree took {fin - st} seconds.")


    st = time.time()
    random.setstate(z3_solve(output))
    for _ in range(number):
        assert output[_][0] == random.getrandbits(bits)
    for _ in range(10000):
        assert check[_] == random.getrandbits(bits)
    fin = time.time()
    print(f"z3 took {fin - st} seconds.")
```

이는 제가 사용한 검증 코드입니다. import한 rbtree\_mersenne 모듈은 rbtree님의 Twister, Solver 클래스를 포함합니다.

둘 다 같은 26비트의 output 1248개를 준 상태에서 풀어내는지 몇 초가 걸리는지를 계산하였습니다.

<br><br>

```
rbtree took 674.332102060318 seconds.
z3 took 9.002115964889526 seconds.
```

z3은 twist 횟수가 적을 경우, 즉 getrandbits에서 아는 비트 수가 많은 경우에는 믿을 수 없을 정도로 빠른 성능을 보입니다.

실험해본 결과 약 15비트 이상의 경우는 30초 내의 빠른 풀이를 보여주었습니다.

<br>

RREF를 이용한 코드는 구현 구조상 꾸준히 10 ~ 12분 사이의 시간이 소요되었습니다.

다음은 getrandbits(2)를 이용한 복구를 진행해보겠습니다.

이러한 상황은 주로 random.random()으로 생성된 값의 범위를 알 때 최상위 비트(들)만을 알 수 있을 때 등장합니다.

예를 들어 math.floor(random.random() \* 6) + 1 의 값으로 주사위 눈의 개수를 결정하는 문제같은 경우에는, 최상위 비트 1 ~ 2개밖에 알 수 없음이 자명합니다.

```
rbtree took 698.5456099510193 seconds.
...
```

예상한 대로 RREF는 10분 가량 걸렸고, z3은 30분이 지나도 풀리지 않아 reasonable time 내에는 풀리기 어렵다는 결론을 내릴 수 있습니다.

상황에 따라서 z3, 또는 직접 행렬방정식을 푸는 것이 각각 유용하다는 것을 알 수 있었습니다.

<br><br>

#### **3\. random.seed**

다시 PythonMT19937 클래스로 돌아가 seed는 어떻게 작동하는 건지 관찰해봅시다. 가끔씩은 나중 값을 예측하는 문제가 아닌, 특정 조건을 맞추는 seed를 구하는 문제들이 등장하기도 하기 때문에 알아두면 도움됩니다.

(SECCON 2022 - janken vs kurenaif, CCE 2022 - Card Sharper 등)

```python
    def seed(self, n):
        lower = 0xffffffff
        keys = []

        while n:
            keys.append(n & lower)
            n >>= 32

        if len(keys) == 0:
            keys.append(0)

        self.init_by_array(keys)

    def init_by_array(self, keys):
        MT19937.seed(self, 0x12bd6aa)
        i, j = 1, 0
        for _ in range(max(624, len(keys))):
            self.mt[i] = self._int32((self.mt[i] ^ ((self.mt[i-1] ^
                            (self.mt[i-1] >> 30)) * 0x19660d)) + keys[j] + j)
            i += 1
            j += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
            j %= len(keys)

        for _ in range(623):
            self.mt[i] = self._int32((self.mt[i] ^ ((self.mt[i-1] ^
                            (self.mt[i-1] >> 30)) * 0x5d588b65)) - i)
            i += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1

        self.mt[0] = 0x80000000
```

시드를 32비트 단위로 잘라서 배열을 만든 후 init\_by\_array 함수를 실행합니다.

시드로부터 mt를 형성하는 과정은 위와 달리 100% 선형적이지만은 않지만 다행히 복잡하지 않은 연산들이라 역연산이 존재합니다. 즉 목표하는 state가 존재한다면 해당하는 seed를 찾을 수 있다는 뜻입니다.

여기서 살짝은 신기하고 유의할 점은 마지막 줄을 보면 초기의 mt\[0\]은 항상 0x80000000으로 설정해주는 것을 확인할 수 있습니다. seed가 등장하는 문제에서는 목표하는 초기 state를 만들 때 constraints로 0x80000000을 수동으로 집어넣어주는 것을 잊지 않도록 합시다.

<br><br>

#### **4\. random의 함수들**

랜덤에서 가장 자주 쓰이는 함수들은 아마 random.random, random.randrange, random.shuffle 등일 것으로 생각됩니다.

random.random은 가장 자주 등장하는 함수로 0 이상 1 미만의 실수값 하나를 생성하는 함수입니다.

이의 구현은 다음과 같이 되어 있습니다.

```python
((random.getrandbits(27) * 67108864.0 + random.getrandbits(26)) * (1.0 / 9007199254740992.0))
```

<br>

즉 실수의 구조는 단순히 53비트의 실수로 구현되어 있고, 먼저 호출한 27비트가 26개의 MSB, 나중에 호출한 26비트가 LSB를 담당합니다.

math.floor(random.random() \* 4) == 2라면, 첫 getrandbits(32)의 상위 비트 2개는 1, 0이고, 둘째 getrandbits(32)에 관한 정보는 알 수 없다고 할 수 있습니다.

<br>

Mersenne Twister은 아니지만 이러한 특징을 이용한 재미있는 문제가 PlaidCTF 2023 - fastrology 시리즈에 출제되었습니다.

SEETF 2022 - Probability 또한 random.random()을 파괴하는 문제고, 이 문제는 메르센 트위스터 말고도 다른 재미있는 단계가 숨어있는 문제라서 정말 강추 드립니다.

random.randrange 함수는 구현을 뜯어보면 결과적으로 random.randbelow를 호출하고, randbelow는 다음과 같이 구현되어 있습니다.

```python
    def _randbelow_with_getrandbits(self, n):
        "Return a random int in the range [0,n).  Returns 0 if n==0."

        if not n:
            return 0
        getrandbits = self.getrandbits
        k = n.bit_length()  # don't use (n-1) here because n can be 1
        r = getrandbits(k)  # 0 <= r < 2**k
        while r >= n:
            r = getrandbits(k)
        return r
```

결과적으로 getrandbits로부터 값을 계산하는 과정을 이해하면 constraints 비트들을 설정하는 데에 큰 문제가 없을 것입니다.

random.shuffle은 다음과 같이 구현되어 있습니다.

```python
            for i in reversed(range(1, len(x))):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = randbelow(i + 1)
                x[i], x[j] = x[j], x[i]
```

randbelow를 이용해 조작할 수 있음을 알 수 있습니다.

CCE 2022 - Card Sharper이 이 random.shuffle과 random.seed를 다룬 문제입니다. 국내 CTF인지라 rbtree님 혼자 솔브를 기록하셨던 것으로 기억합니다.

<br><br>

## **JavaScript's random**

PlaidCTF 2023에 출제된 fastrology 시리즈는 javascript에서 사용되는 random 모듈에서 값 수집 후 나중 값을 예측하는 문제들이 출제되었습니다.

javascript의 random에서 사용하는 xorshift128은 다음과 같이 구현되어 있습니다.

```js
uint64_t state0, state1;    // the 128-bit state

// update state and produce a double
double MathRandom(void) {
  uint64_t s0 = state1;  // notice the swap
  uint64_t s1 = state0;
  s1 = s1 ^ (s1 << 23);
  s1 = s1 ^ (s1 >> 17) ^ (s0 >> 26) ^ s0;
  state0 = s0;
  state1 = s1;
  if (Firefox || Webkit) // Firefox v47.0, Webkit 2019-07 thru 2020-02
    return (double)( (s0+s1) & (((uint64_t)1<<53)-1) ) / ((uint64_t)1<<53);
  else                   // ChromX V8, 2019-01 thru 2020-02
    return (double)(  s0     >> 12                   ) / ((uint64_t)1<<52);
}
```

얼핏 봐도 구조가 Mersenne Twister보다는 훨씬 간단하고, state 업데이트 또한 선형적으로 이루어지는 것을 확인할 수 있습니다. 단, javascript의 random이 특이한 점은 브라우저별로 사용하는 랜덤 알고리즘이 다르다는 것이었습니다.

코드의 밑쪽을 봐보면 s0 >> 12를 사용하는 경우가 있고, s0 + s1 을 사용하는 경우가 존재합니다. s0 >> 12는 선형성이 유지되기 때문에 결과를 이용해서 초기 s0, s1을 쉽게 계산할 수 있지만, 64비트 덧셈은 비트 기준으로 선형적이지 않기 때문에 s0 + s1이 등장하는 순간 문제가 이전처럼 쉽지만은 않아집니다.

<br>

다행히도 fastrology에서는 s0 >> 12를 채택하였고, 아까 살짝 언급한 MSB들 2 ~ 3개를 이용한 행렬식 형성을 이용해 풀 수 있는 재미있는 문제였습니다.

<br><br>

#### **참고 문헌**

1\. random.py - [https://github.com/python/cpython/blob/main/Lib/random.py](https://github.com/python/cpython/blob/main/Lib/random.py)

2\. Mersenne Twister Pseudocode - [https://github.com/eboda/mersenne-twister-recover/blob/master/MersenneTwister.py](https://github.com/eboda/mersenne-twister-recover/blob/master/MersenneTwister.py)

3\. Mersenne Twister Wikipedia - [https://en.wikipedia.org/wiki/Mersenne\_Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)

4\. Breaking Python Random Module - [https://rbtree.blog/posts/2021-05-18-breaking-python-random-module/](https://rbtree.blog/posts/2021-05-18-breaking-python-random-module/)

5\. Breaking Python Random with z3 - [https://github.com/cryptohack/ctf\_archive/blob/main/SEETF2022\_probability/server\_files/solve.ipynb](https://github.com/cryptohack/ctf_archive/blob/main/SEETF2022_probability/server_files/solve.ipynb)