import itertools
import numpy as np

def find_max_2d_array(array):
    max_value = float('-inf')  # Initialize with the smallest possible value
    for row in array:
        for element in row:
            if element > max_value:
                max_value = element
    return max_value

def find_min_2d_array(array):
    min_value = float('inf')  # Initialize with the smallest possible value
    for row in array:
        for element in row:
            if element < min_value:
                min_value = element
    return min_value

def are_collinear(p1, p2, p3):
    # Check if the area of the triangle formed by the points is zero
    # Using the determinant of the matrix to calculate the area
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) == 0

def ToHypergraph(n):
    points = [[x, y] for x in range(n) for y in range(n)]
    degree = [[0 for x in range(n)] for y in range(n)]
    collinear_triples = []

    for p1, p2, p3 in itertools.combinations(points, 3):
        if are_collinear(p1, p2, p3):
            collinear_triples.append((p1, p2, p3))
            degree[p1[0]][p1[1]]+=1
            degree[p2[0]][p2[1]]+=1
            degree[p3[0]][p3[1]]+=1

    return [points, collinear_triples, degree]


n = 47 # Change this value for a different grid size
#Hypergraph = ToHypergraph(n)
Hypergraph = np.load("datas//case"+str(n)+".npy", allow_pickle=True)
#np.save('datas\case'+str(n), Hypergraph)
print(f"Collinear triples in a {n}x{n} grid:")
#for triple in Hypergraph[1]:
    #print(triple)

points = Hypergraph[0]
collinear_triple = Hypergraph[1]
degree = Hypergraph[2]
collinear_triple_num = len(Hypergraph[1])
max_degree = find_max_2d_array(degree)
min_degree = find_min_2d_array(degree)
print("collinear triple 개수 : "+str(collinear_triple_num))
print("max-degree : "+str(max_degree))
print("min-degree : "+str(min_degree))
hittingset_lowerbound_lemma58 = (n*n)/((max_degree)+1)
hittingset_lowerbound_lemma58_plus = (n*n)/((2*max_degree/min_degree)+1)
print("upper-bound : "+str(n*n-hittingset_lowerbound_lemma58_plus)) #모두 2*n보다 매우 큼...ㅠ
print("lowerr-bound : "+str(4*n*n/(max_degree+5))) #모두 2*n보다 매우 큼...ㅠ