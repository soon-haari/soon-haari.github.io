R = RealField(100)

a = R(e) ^ (R(pi) * i / R(5))
b = a^2

coefs = [1/(a^24 - a^22 - a^20 + 2*a^14 - a^8 - a^6 + a^4), 1/(-a^24 + 2*a^22 - a^18 - a^16 + 2*a^12 - a^10), (-1)/(-a^26 + 2*a^24 + a^22 - 4*a^20 + a^18 + 2*a^16 - a^14), 1/(-a^30 + 2*a^28 - a^24 - a^22 + 2*a^18 - a^16), (-1)/(-a^36 + a^34 + a^32 - 2*a^26 + a^20 + a^18 - a^16)]

A, B = 0, 1

ans = 0

for i in range(5):
	ans += coefs[i] * (ln(B - a * b^i) - ln(A - a * b^i))

print(ans)