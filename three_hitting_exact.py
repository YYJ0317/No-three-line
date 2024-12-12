import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def is_simple(H):
    """
    Check if the hypergraph H is simple.
    A hypergraph is simple if no edge is a subset of another edge.
    """
    for i in range(len(H)):
        for j in range(len(H)):
            if i != j and set(H[i]).issubset(set(H[j])):
                return False
    return True

def minimize_hypergraph(H):
    """
    Minimize the hypergraph by removing edges that are supersets of other edges and removing duplicates.
    """
    minimal_H = []
    for edge in H:
        if not any(set(edge).issuperset(set(other)) for other in H if edge != other):
            minimal_H.append(edge)
    # Remove duplicates
    minimal_H = [tuple(sorted(e)) for e in minimal_H]  # Convert to sorted tuples
    minimal_H = list(set(minimal_H))  # Remove duplicates
    return [list(e) for e in minimal_H]  # Convert back to lists


def connected_components(H):
    """
    Find the connected components of the hypergraph H.
    """
    G = nx.Graph()
    for edge in H:
        edge_list = list(edge)
        G.add_nodes_from(edge_list)
        G.add_edges_from([(edge_list[i], edge_list[j]) for i in range(len(edge_list)) for j in range(i + 1, len(edge_list))])
    components = list(nx.connected_components(G))
    component_edges = []
    for component in components:
        component_edges.append([edge for edge in H if any(v in edge for v in component)])
    return component_edges

def vertex_degree(H):
    """
    Compute the degree of each vertex in the hypergraph H.
    """
    degree = {}
    for edge in H:
        for vertex in edge:
            if vertex not in degree:
                degree[vertex] = 0
            degree[vertex] += 1
    return degree

def d2_vertex_degree(H):
    """
    Compute the degree 2 (d2) of each vertex in the hypergraph H.
    """
    d2 = {}
    for edge in H:
        if len(edge) == 2:
            for vertex in edge:
                if vertex not in d2:
                    d2[vertex] = 0
                d2[vertex] += 1
    return d2

def remove_vertex(H, x):
    """
    Remove vertex x from every edge in H.
    """
    return [set(edge).difference({x}) for edge in H]

def exclude_vertex(H, x):
    """
    Remove all edges containing vertex x from H.
    """
    return [edge for edge in H if x not in edge]

def min_tr(H, k):
    #print(H)
    """
    Algorithm 59: MinTr
    Finds a hitting set of size at most k for the hypergraph H.
    """
    # 0. If H is empty, then return ∅.
    if not H:
        print("case : 0")
        return set()
    
    # 1. If k = 0, then return V(H).
    if k == 0:
        print("case : 1")
        return set().union(*H)
    
    # 2. If H is not simple, then return MinTr(Min(H), k).
    if not is_simple(H):
        print("case : 2")
        return min_tr(minimize_hypergraph(H), k)
    
    # 3. If H consists of connected components C1, ..., Ct, then return ⋃i MinTr(Ci, ki).
    components = connected_components(H)
    if len(components) > 1:
        print("case : 3")
        print("component 발생!")
        t = len(components)
        k_values = [k - (t - 1)]  # Initialize k1
        cards = []
        result = set()
        for i, component in enumerate(components):
            if i > 0:
                k_values.append(k_values[-1] - cards[-1] + 1)
            component_hitting_set = min_tr(component, k_values[-1])
            cards.append(len(component_hitting_set))
            result.update(component_hitting_set)
        return result
    
    # 4. If there exists a loop {x}, then return {x} ∪ MinTr(H[x=1], k - 1).
    for edge in H:
        if len(edge) == 1:
            print("case : 4")
            x = next(iter(edge))
            reduced_H = exclude_vertex(H, x)
            return {x}.union(min_tr(reduced_H, k - 1))

    # 5. If some vertex x is dominated by some other vertex, then return MinTr(H[x=0], k).
    degree = vertex_degree(H)
    d_H = max(degree.values())
    for x in degree:
        for y in degree:
            if x != y and all((x in edge and y in edge) for edge in H if x in edge):
                print("case : 5, "+str(y)+" dominates "+str(x))
                reduced_H = remove_vertex(H, x)
                return min_tr(reduced_H, k)

    # 6. If H is 3-uniform, then let d = d(H). If k > n(H) - (d + 1) / (d + 5), then return MinTr(H, ⌊n(H) - (d + 1) / (d + 5)⌋). If k < n(H) / (d + 1), then return V(H).
    if all(len(e) == 3 for e in H):  # Check if H is 3-uniform
        print("case : 6")
        n_H = len(set().union(*H))
        if k > n_H - (d_H + 1) / (d_H + 5):
            k = int(n_H - (d_H + 1) / (d_H + 5))
        if k < n_H / (d_H + 1):
            return set().union(*H)

    # 7. If there exists some 2-vertex x involved in edges E1 and E2, and there exists some edge E ⊆ (E1 ∪ E2) \ {x}, then return MinTr(H[x=0], k).
    for x, deg in degree.items():
        if deg == 2:
            edges_with_x = [e for e in H if x in e]
            #if len(edges_with_x) == 2:
            E1, E2 = edges_with_x
            if any(set(e).issubset((set(E1).union(E2)).difference({x})) for e in H if e != E1 and e != E2):
                print("case : 7")
                reduced_H = remove_vertex(H, x)
                return min_tr(reduced_H, k)

    # 8. If there exists some vertex v with d2(v) > 0 and d(v) ≤ 3, then let x be a vertex with maximum d(x) among all vertices with maximum d2(x). If d2(x) ≥ 1 and d(x) ≤ 3, then return min({x} ∪ MinTr(H[x=1], k - 1), MinTr(H[x=0], k)).
    d2 = d2_vertex_degree(H)
    candidates = [v for v, deg in degree.items() if deg >= 3 and d2.get(v, 0) > 0]
    if candidates:
        print("case : 8")
        x = max(candidates, key=lambda v: (d2[v], degree[v]))
        reduced_H_x1 = exclude_vertex(H, x)
        hitting_set_x1 = {x}.union(min_tr(reduced_H_x1, k - 1))
        reduced_H_x0 = remove_vertex(H, x)
        hitting_set_x0 = min_tr(reduced_H_x0, k)
        return hitting_set_x1 if len(hitting_set_x1) < len(hitting_set_x0) else hitting_set_x0

    # 9. If there exists some 2-vertex v with d2(v) ≥ 1, then let x be a vertex that maximizes d2(x) and let E1, E2 be the edges containing x. Assuming |E1| ≤ |E2|, let E1 = {x, y}. If |E2| = 2, then let E2 = {x, z}, and return min({x} ∪ MinTr(H[x=1, y=z=0], k - 1), {y, z} ∪ MinTr(H[x=0, y=z=1], k - 2)). Otherwise, let E2 = {x, z, w}. Return min({x} ∪ MinTr(H[x=1, y=z=w=0], k - 1), {y} ∪ MinTr(H[x=0, y=1], k - 1)).
    candidates = [v for v, d2v in d2.items() if d2v >= 1 and len([e for e in H if v in e]) == 2]
    if candidates:
        x = max(candidates, key=d2.get)
        print(x)
        edges_with_x = [e for e in H if x in e]
        if len(edges_with_x) == 2:
            E1, E2 = sorted(edges_with_x[:2], key=len)
            E1_list, E2_list = list(E1), list(E2)
            if len(E1_list) == 2 :
                print("case : 9")
                if len(E2_list) == 2:
                    _, y = E1_list
                    _, z = E2_list
                    print(str(y)+"와"+str(z))
                    reduced_H_x1 = exclude_vertex(H, x)
                    reduced_H_x1_yz0 = remove_vertex(reduced_H_x1, y)
                    reduced_H_x1_yz0 = remove_vertex(reduced_H_x1_yz0, z)
                    hitting_set_x1 = set({x}).union(min_tr(reduced_H_x1_yz0, k - 1))
                    print(hitting_set_x1)
                    print("!!!!!!!!!!!!!")
                    reduced_H_x0 = remove_vertex(H, x)
                    reduced_H_x0_yz1 = exclude_vertex(reduced_H_x0, y)
                    reduced_H_x0_yz1 = exclude_vertex(reduced_H_x0_yz1, z)
                    hitting_set_yz = set({y, z}).union(min_tr(reduced_H_x0_yz1, k - 2))
                    print(hitting_set_yz)
                    print("!!!!!!!1")
                    return hitting_set_x1 if len(hitting_set_x1) < len(hitting_set_yz) else hitting_set_yz
                else:
                    _, y = E1_list
                    _, z, w = E2_list
                    reduced_H_x1 = exclude_vertex(H, x)
                    reduced_H_x1_yzw0 = remove_vertex(reduced_H_x1, y)
                    reduced_H_x1_yzw0 = remove_vertex(reduced_H_x1_yzw0, z)
                    reduced_H_x1_yzw0 = remove_vertex(reduced_H_x1_yzw0, w)
                    hitting_set_x1 = {x}.union(min_tr(reduced_H_x1_yzw0, k - 1))
                    reduced_H_x0 = remove_vertex(H, x)
                    reduced_H_x0_y1 = exclude_vertex(reduced_H_x0, y)
                    hitting_set_y = {y}.union(min_tr(reduced_H_x0_y1, k - 1))
                    return hitting_set_x1 if len(hitting_set_x1) < len(hitting_set_y) else hitting_set_y

    # 10. If d(H) ≤ 3 and d(v)=2 for some vertex v, then assume that the edges containing v are {v, w, x}, {v, y, z}. Return min({x} ∪ MinTr(H[v=1, w=x=y=z=0], k - 1), MinTr(H[x=0], k)).
    if d_H <= 3:
        for v, deg in degree.items():
            if deg == 2:
                print("case : 10")
                edges_with_v = [e for e in H if v in e]
                #if len(edges_with_v) == 2:
                E1, E2 = edges_with_v
                E1_list, E2_list = list(E1), list(E2)
                #if len(E1_list) == 3:
                v, w, x = E1_list
                v, y, z = E2_list
                reduced_H_v1 = exclude_vertex(H, v)
                reduced_H_v1 = remove_vertex(reduced_H_v1, w)
                reduced_H_v1 = remove_vertex(reduced_H_v1, x)
                reduced_H_v1 = remove_vertex(reduced_H_v1, y)
                reduced_H_v1 = remove_vertex(reduced_H_v1, z)
                hitting_set_v1 = {x}.union(min_tr(reduced_H_v1, k - 1))
                reduced_H_v0 = remove_vertex(H, v)
                hitting_set_v0 = min_tr(reduced_H_v0, k)
                return hitting_set_v1 if len(hitting_set_v1) < len(hitting_set_v0) else hitting_set_v0

    # 11. Finally, pick a vertex x with maximum d(x) and return min({x} ∪ MinTr(H[x=1], k - 1), MinTr(H[x=0], k)).
    print("case : 11")
    x = max(degree, key=degree.get)
    reduced_H_x1 = exclude_vertex(H, x)
    hitting_set_x1 = set({x}).union(min_tr(reduced_H_x1, k - 1))
    reduced_H_x0 = remove_vertex(H, x)
    hitting_set_x0 = min_tr(reduced_H_x0, k)
    return hitting_set_x1 if len(hitting_set_x1) < len(hitting_set_x0) else hitting_set_x0

import time

def greedy_3_hitting_set(sets):
    # Initialize an empty hitting set
    hitting_set = set()

    # Copy the sets to avoid modifying the original input
    remaining_sets = [s.copy() for s in sets]

    # Calculate the frequency of each element
    element_frequency = {}
    for s in remaining_sets:
        for element in s:
            if element in element_frequency:
                element_frequency[element] += 1
            else:
                element_frequency[element] = 1

    iteration = 0
    time_sum = 0
    while remaining_sets:
        iteration += 1
        start_time = time.time()

        # Find the element with the maximum frequency
        max_element = max(element_frequency, key=element_frequency.get)

        # Add this element to the hitting set
        hitting_set.add(max_element)

        # Remove all sets that contain this element
        new_remaining_sets = []
        for s in remaining_sets:
            if max_element not in s:
                new_remaining_sets.append(s)
            else:
                for element in s: element_frequency[element] -= 1
        
        remaining_sets = new_remaining_sets.copy()

        end_time = time.time()
        print(f"Iteration {iteration}: Time consumed = {end_time - start_time:.6f} seconds")
        time_sum += end_time - start_time

    print(f"Total time : {time_sum}")
    return hitting_set

def drawing(n, hitting_set):
    
    hitting_set = sorted(list(hitting_set))
    len_hit = len(hitting_set)
    idx = 0
    x_points = []
    y_points = []

    for i in range(n):
        for j in range(n):
            if idx < len_hit and n*i+j == hitting_set[idx] : 
                idx += 1
            else : 
                x_points.append(j+0.5)
                y_points.append(i+0.5)

    fig, ax = plt.subplots()

    # Set the limits of the plot
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # Set grid
    ax.grid(True)

    # Draw points
    ax.scatter(x_points, y_points, s=100, color='black')

    # Customize grid lines to match the style
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Customize minor grid lines to make them more hand-drawn style
    ax.grid(which='minor', linestyle='-', linewidth='1')

    plt.title(str(n*n-len(hitting_set))+" / "+str(2*n)+"")
    plt.show()

n = 3
k = 1

H = np.load("datas\\case"+str(n)+".npy", allow_pickle=True)
for i in range(len(H[1])):
    H[1][i] = list(H[1][i])
    for j in range(3):
        H[1][i][j] = n*H[1][i][j][1]+H[1][i][j][0]

#hitting_set = min_tr(H[1], k)
hitting_set = greedy_3_hitting_set(H[1])

print("Minimum Hitting Set:", hitting_set)
drawing(n, hitting_set)






