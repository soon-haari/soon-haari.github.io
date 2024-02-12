import os

def state2seed(state):

    N = 624
    uint32_mask = 1 << 32
    state = list(state[1][:-1])
    state[0] = state[N-1]
    i = 2
    for k in range(1,N):
        i = i - 1
        state[i] = ((state[i] + i) ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941)) % uint32_mask
        if i == 1:
            i = N

    state[0] = state[N-1]

    origin_state = [0 for _ in range(N)]
    origin_state[0] = 19650218
    for i in range(1,N):
        origin_state[i] = (1812433253 * (origin_state[i-1] ^ (origin_state[i-1] >> 30)) + i) % uint32_mask

    key = [0 for i in range(N)]

    key[0] = 1 #no problem

    origin_state[1] = ((origin_state[1] ^ ((origin_state[0] ^ (origin_state[0] >> 30)) * 1664525)) + key[0] + 0) % uint32_mask

    i = 2
    j = 1
    k = N - 1

    while(k):
        x = ((origin_state[i] ^ ((origin_state[i-1] ^ (origin_state[i-1] >> 30)) * 1664525)) + j) % uint32_mask
        key[j] = (state[i] - x) % uint32_mask
        origin_state[i] = ((origin_state[i] ^ ((origin_state[i-1] ^ (origin_state[i-1] >> 30)) * 1664525)) + key[j] + j) % uint32_mask

        i += 1
        if i == N:
            origin_state[0] = origin_state[N-1]
            i = 1
        j += 1
        k -= 1

    mySeed = 0
    for i in range(N-1,-1,-1):
        mySeed = mySeed << 32
        mySeed += key[i]
    return mySeed

def check():
    import random
    import time
    N = 1000
    for _ in range(10):
        random.seed(random.randint(0, 2**19968))
        state = random.getstate()
        print(state)
        k = random.sample(range(8501), 2125)

        seed = state2seed(state)
        random.seed(seed)

        state = random.getstate()
        print(state)
        kk = random.sample(range(8501), 2125)

        # print(k[:10])
        # print(kk[:10])
        assert k == kk

        break

if __name__ == "__main__":
    check()
    print("check : success")