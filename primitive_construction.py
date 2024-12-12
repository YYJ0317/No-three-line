import matplotlib.pyplot as plt
import numpy as np

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

def find_primitive_root_and_powers(p):
    if not is_prime(p):
        return "Input must be a prime number"
    
    phi = p - 1
    factors = prime_factors(phi)
    
    for g in range(2, p):
        flag = True
        for q in factors:
            if pow(g, phi // q, p) == 1:
                flag = False
                break
        if flag:
            powers_g = [pow(g, i, p) for i in range(1, p)]
            powers_p_minus_g = [pow(p - g, i, p) for i in range(1, p)]
            return g, powers_g, powers_p_minus_g

def drawing(n, y1_points, y2_points):
    
    x_points = [i for i in range(1,n+1)]
 
    fig, ax = plt.subplots()

    # Set the limits of the plot
    ax.set_xlim(1, n+1)
    ax.set_ylim(1, n+1)

    # Set grid
    ax.grid(True)

    # Draw points
    ax.scatter([x+0.5 for x in x_points], [y+0.5 for y in y1_points], s=10/n, color='black')
    #ax.scatter([x+0.5 for x in x_points], [y+0.5 for y in y2_points], s=10/n, color='black')

    # Customize grid lines to match the style
    ax.set_xticks(np.arange(1, n+1, 1))
    ax.set_yticks(np.arange(1, n+1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Customize minor grid lines to make them more hand-drawn style
    ax.grid(which='minor', linestyle='-', linewidth='1')

    plt.show()

# Example usage:
p = 13
primitive_root, powers_list, powers_p_minus_list = find_primitive_root_and_powers(p)
print(f"A primitive root of {p} is {primitive_root}")
print(f"List of powers of {primitive_root}: {powers_list}")
print(f"List of powers of {p} - {primitive_root}: {powers_p_minus_list}")
drawing(p-1, powers_list, powers_p_minus_list)