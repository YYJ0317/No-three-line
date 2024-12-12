import math
import numpy as np
import matplotlib.pyplot as plt

#################################################################################################

def S(n):
    """Generate the Farey sequence of order n."""
    farey_list = []
    a, b, c, d = 0, 1, 1, n  # Initialize the first two fractions 0/1 and 1/n

    farey_list.append((a, b-a))  # Append 0/1

    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        farey_list.append((a, b-a))

    x = []
    y = []
    for element in farey_list:
        if element[0]==0 or element[1]==0 : continue
        x.append(element[1])
        y.append(element[0])

    return [x,y]


def m(t):
    """
    Calculate the sum of all p where gcd(p, q) = 1 and p + q <= t.
    :param t: A natural number
    :return: The value of m(t)
    """
    total_sum = 0
    for p in range(1, t):
        for q in range(1, t):
            if p + q <= t and math.gcd(p, q) == 1:
                total_sum += p
    
    return 1 + 2 * total_sum

def generate_sequence(a, b, m_t):
    """Generate sequence A and the additional points based on the conditions described in the image."""
    h = len(a)
    A = []
    points = []
    
    # Initialize A(0)
    A.append(((m_t + 1) // 2, 0))
    
    # Generate the sequence A
    for i in range(1, h + 1):
        x_i, y_i = A[i-1]
        A.append((x_i + a[i-1], y_i + b[i-1]))
    
    print(A)

    # Generate the additional points based on A
    for i in range(h + 1):
        x_i, y_i = A[i]
        points.append((x_i,y_i))
        points.append((m_t - x_i, y_i))
        points.append((x_i, m_t - y_i))
        points.append((m_t - x_i, m_t - y_i))
    
    return points

#################################################################################################

def are_collinear(p1, p2, p3):
    # Check if the area of the triangle formed by the points is zero
    # Using the determinant of the matrix to calculate the area
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) == 0

def filter_collinear(points, chosen):
    """
    Filter out points from the points list that are collinear with any two points in the chosen list.
    :param points: List of tuples [(x1, y1), (x2, y2), ...]
    :param chosen: List of tuples [(xc1, yc1), (xc2, yc2), ...]
    :return: List of tuples, points that are not collinear with any pair of points in chosen
    """
    non_collinear_points = []
    chosen_pairs = [(chosen[i], chosen[j]) for i in range(len(chosen)) for j in range(i + 1, len(chosen))]

    for point in points:
        collinear = False
        for pair in chosen_pairs:
            if are_collinear(pair[0], pair[1], point):
                collinear = True
                break
        if not collinear:
            non_collinear_points.append(point)
    
    return non_collinear_points


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]

#################################################################################################

def drawing(points, n):
 
    fig, ax = plt.subplots()

    # Set the limits of the plot
    ax.set_xlim(0, n+1)
    ax.set_ylim(0, n+1)

    # Set grid
    ax.grid(True)

    # Draw points
    ax.scatter([point[0]+0.5 for point in points], [point[1]+0.5 for point in points], s=10/n, color='black')

    # Customize grid lines to match the style
    ax.set_xticks(np.arange(0, n+1, 1))
    ax.set_yticks(np.arange(0, n+1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Customize minor grid lines to make them more hand-drawn style
    ax.grid(which='minor', linestyle='-', linewidth='2')
    plt.title("n = "+str(n+1)+", # = "+str(len(points)))
    plt.show()

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

###################################################################################################

# Example usage
t = 4
sequence = S(t)
m_t = m(t)
print("m({t})+1 : "+str(m_t+1))
print("Special sequence of order", t, "is:", sequence)
points = generate_sequence(sequence[0],sequence[1],m_t)
drawing(points,m_t)

all = []
for x in range(m_t):
    for y in range(m_t):
        all.append((x, y))
all = filter_collinear(all, points)

while all:
    hull = convex_hull(all)
    all = filter_collinear(all, points)
    drawing(all, m_t)
    print(len(all))
    points += hull

points = remove_duplicates(points)

print("convex greedy points : "+str(len(points)))
print(points)
drawing(points, m_t)