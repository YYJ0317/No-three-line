def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def find_primitive_roots(p):
    if not is_prime(p):
        raise ValueError("Input must be a prime number")
    
    phi = p - 1
    factors = prime_factors(phi)
    primitive_roots = []
    
    for g in range(2, p):
        flag = True
        for q in factors:
            if pow(g, phi // q, p) == 1:
                flag = False
                break
        if flag:
            primitive_roots.append(g)
    
    return primitive_roots

def check_equation_for_primitive_root(p, r):
    for t in range(1, p - 1):
        for s in range(1, p - 1 - t):
            left_side = t * pow(r, t, p) * (pow(r, s, p) - 1) % p
            right_side = s * (pow(r, t, p) - 1) % p
            if left_side == right_side:
                return False, (s, t)
    return True, None

def find_primitive_roots_satisfying_equation(p):
    primitive_roots = find_primitive_roots(p)
    solutions = []
    
    for r in primitive_roots:
        result, st_pair = check_equation_for_primitive_root(p, r)
        if result:
            solutions.append((r, "True"))
        else:
            solutions.append((r, "False", st_pair))
    
    return solutions

# Example usage:
p = 47
solutions = find_primitive_roots_satisfying_equation(p)
for solution in solutions:
    print(solution)
