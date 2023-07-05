# Introduction - Finding Flags

    Each challenge is designed to help introduce you to a new piece of cryptography. Solving a challenge will require you to find a "flag".

    These flags will usually be in the format crypto{y0ur_f1rst_fl4g}. The flag format helps you verify that you found the correct solution.

    Try submitting this into the form below to solve your first challenge.

아주 친절한 문제다.

flag

    crypto{y0ur_f1rst_fl4g}

</br></br></br>

# Introduction - Great Snakes

    #!/usr/bin/env python3

    import sys
    # import this

    if sys.version_info.major == 2:
        print("You are running Python 2, which is no longer supported. Please update to Python 3.")

    ords = [81, 64, 75, 66, 70, 93, 73, 72, 1, 92, 109, 2, 84, 109, 66, 75, 70, 90, 2, 92, 79]

    print("Here is your flag:")
    print("".join(chr(o ^ 0x32) for o in ords))

실행시키면 플래그를 준다.

flag

    crypto{z3n_0f_pyth0n}

</br></br></br>

# Introduction - Network Attacks

    Several of the challenges are dynamic and require you to talk to our challenge servers over the network. This allows you to perform man-in-the-middle attacks on people trying to communicate, or directly attack a vulnerable service. To keep things consistent, our interactive servers always send and receive JSON objects. 

    Python makes such network communication easy with the telnetlib module. Conveniently, it's part of Python's standard library, so let's use it for now.

    For this challenge, connect to socket.cryptohack.org on port 11112. Send a JSON object with the key buy and value flag.

    The example script below contains the beginnings of a solution for you to modify, and you can reuse it for later challenges.

    Connect at nc socket.cryptohack.org 11112

buy flag를 json 형식으로 입력해주면 된다. 

    {"buy": "flag"}

flag

    crypto{sh0pp1ng_f0r_fl4g5}