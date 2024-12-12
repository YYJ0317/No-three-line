import matplotlib.pyplot as plt
import numpy as np

def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

def drawing(n, y_points):
    
    x_points = [i for i in range(n)]
 
    fig, ax = plt.subplots()

    # Set the limits of the plot
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # Set grid
    ax.grid(True)

    # Draw points
    ax.scatter([x+0.5 for x in x_points], [y+0.5 for y in y_points], s=100, color='black')

    # Customize grid lines to match the style
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Customize minor grid lines to make them more hand-drawn style
    ax.grid(which='minor', linestyle='-', linewidth='1')
    plt.title("Hall construction, p = "+str(p)+", k = "+str(k))
    plt.show()

def construct_list_y(p, k):
    # Ensure that p is a prime number
    if p < 2:
        raise ValueError("p must be a prime number greater than 1.")
    
    # Function to check if a number is prime
    
    #if not is_prime(p):
        raise ValueError("p must be a prime number.")
    
    y_list = []
    check = 0 
    for x in range(p):
        # Find y such that (xy - k) % p == 0
        for y in range(p):
            if (x * y - k) % p == 0:
                check = 1
                y_list.append(y)
                break  # No need to find another y for the same x
        if check == 0 : y_list.append(-1)
        check = 0
    
    drawing(p, y_list)
    return y_list

# Example usage:
p = 15  
k = 7  # An integer
result = construct_list_y(p, k)
print(f"The list of y for p={p} and k={k} is: {result}")