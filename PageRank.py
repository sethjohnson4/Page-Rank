import numpy as np
import networkx as nx
import scipy as sp
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import time

'''
This python code calculates the top web pages for the google page rank algorithm.
It inputs in a .txt file which contains a from and destination node representing web page links.
The file is read and a directed graph is built. After a directed graph is build a sparse matrix is created. 
The matrix is normalized so that each row adds up to 1, each node has value 1/n.
The power method finds the eigen values of matrix A and gives the dominant value, which is the most resourceful webpage.
The Jacobi and Gauss-Seidel methods focus on the google matrix equation as a more direct approach.
The google matrix equation is then r=αAr+(1−α)v, with r as the PageRank vector,
A is the adjacency matrix (the sparse matrix created), α is the damping factor =0.85,
and v is the teleportation vector (uniform distribution). 
I compute the results with the power method, and then 

I have two files for this. 'web-Google.txt' and 'web-graph.txt', the google web graph
is from kaggle and very large. It is very computationally intensive and may result in errors.
the web graph is a smaller graph, of size 100 which computes quickly and has very readable results.

The link for the google graph: https://www.kaggle.com/datasets/pappukrjha/google-web-graph
'''

def load_graph(file_path):
    """Loads a directed graph from a file and returns a NetworkX DiGraph."""
    print("Loading graph...")
    edges = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            source, target = map(int, line.split())
            edges.append((source, target))

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def build_sparse_matrix(G):
    """Converts a directed graph into a sparse row-stochastic adjacency matrix."""
    print("Building Sparse Matrix...")
    nodes = sorted(G.nodes())
    N = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}  # Map node ID to index

    row, col = [], []
    for source, target in G.edges():
        row.append(node_index[source])  # Source (outgoing link)
        col.append(node_index[target])  # Target (receives link)

    # Create sparse adjacency matrix
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(N, N))

    # Normalize rows to make it stochastic
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero for dangling nodes
    A = A.multiply(1 / row_sums[:, np.newaxis])  # Normalize each row

    print("Matrix size: ", A.shape)
    return A


def power_method(A, num_iterations=1000, tolerance=1e-6):
    """Computes the dominant eigenvector using the Power Method."""
    N = A.shape[0]

    # Initialize with a random vector instead of uniform values
    x = np.random.rand(N)
    x /= np.linalg.norm(x, np.inf)  # Normalize using max norm

    for k in range(num_iterations):
        # Compute y = A * x
        y = A.dot(x)

        # Find p: the index of the largest absolute value in y
        p = np.argmax(np.abs(y))

        # Compute dominant eigenvalue estimate
        mu = y[p]

        # Normalize x using y[p]
        y /= mu

        # Compute error
        err = np.linalg.norm(x - y, np.inf)

        # Check for convergence
        if err < tolerance:
            print(f"Power Method converged in {k + 1} iterations.")
            return y

        x = y  # Update x for next iteration

    print("Power Method did not converge.")
    return x


def page_rank_jacobi(G, alpha=0.85, tol=1e-6, max_iter=500):
    """Computes PageRank using the Jacobi iterative method with sparse matrices."""
    N = G.shape[0]  # Number of nodes
    v = np.ones(N) / N  # Uniform teleportation vector
    b = (1 - alpha) * v  # Right-hand side

    # Compute sparse A = I - alpha * G
    A = identity(N, format="csr") - alpha * G  # Keeps A sparse

    # Extract diagonal elements (sparse format)
    D = A.diagonal()  # This is a dense vector but necessary

    # Extract off-diagonal part of A (keeps it sparse)
    R = A - diags(D, format="csr")  # Keeps R sparse

    # Initialize rank vector
    r = np.ones(N) / N  # Start with uniform probability

    # Jacobi iteration
    for iteration in range(max_iter):
        r_new = (b - R.dot(r)) / D  # Only sparse operations!
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"Jacobi method converged in {iteration} iterations.")
            return r / np.sum(r)
        r = r_new  # Update
    print("Maximum iterations met for jacobi method") # Exit if max iterations hit and return the result
    return r / np.sum(r)  # Normalize final PageRank scores

def page_rank_gauss_seidel(G, alpha=0.85, tol=1e-6, max_iter=500):
    """Computes PageRank using the Gauss-Seidel iterative method"""
    N = G.shape[0]  # Number of nodes
    v = np.ones(N) / N  # Uniform teleportation vector
    b = (1 - alpha) * v  # Right-hand side

    # Compute sparse A = I - alpha * G
    A = identity(N, format="csr") - alpha * G  # Keeps A sparse

    # Extract diagonal elements and compute inverse
    D_inv = diags(1.0 / A.diagonal(), format="csr")  # Sparse diagonal matrix

    # Extract off-diagonal part of A
    R = A - diags(A.diagonal(), format="csr")  # Keep R sparse

    # Initialize rank vector
    r = np.ones(N) / N  # Start with uniform probability

    # Gauss-Seidel iteration (in-place updates)
    for iteration in range(max_iter):
        r_old = r.copy()  # Store old values for convergence check
        r = D_inv.dot(b - R.dot(r))  # Fully sparse computation

        if np.linalg.norm(r - r_old, 1) < tol:
            print(f"Gauss-Seidel method converged in {iteration+1} iterations.")
            return r / np.sum(r)  # Normalize final PageRank scores

    print("Maximum iterations met for Gauss-Seidel method")
    return r / np.sum(r)  # Normalize final PageRank scores


# Run the functions
# web-Google.txt is much larger and very computationally expensive
#file_path = "web-Google.txt"
file_path = "web_graph.txt"

G = load_graph(file_path)
A = build_sparse_matrix(G)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print("Sparse adjacency matrix (stochastic form) created.")

# Run power method on the adjacency matrix A
power_start_time = time.time()
p = power_method(A)
power_end_time = time.time()
print("Page Rank Power Method Complete")

# Run Jacobi method and time the results
jacobi_start_time = time.time()
j = page_rank_jacobi(A)
jacobi_end_time = time.time()
print("PageRank Jacobi Complete")

#run Gauss-Seidel
Gauss_start = time.time()
gs = page_rank_gauss_seidel(A)
Gauss_end_time = time.time()
print("PageRank Gauss-Seidel Complete")

print("\n","="*50,"\n", " "*20,"Results")
print("-"*50)
# Sort and display results for Power Method
print(f"\nPower Method took {power_end_time - power_start_time} seconds")
sorted_indices_p = sorted(range(len(p)), key=lambda i: p[i], reverse=True)
print("\nTop 10 sorted nodes by importance (PageRank) for Power Method:")
for i in sorted_indices_p[:10]:
    print(f"Node {i} has PageRank score {p[i]}")

print("\n","-"*50)
# Sort and display results for Jacobi Method
print(f"\nJacobi Method took {jacobi_end_time - jacobi_start_time} seconds")
sorted_indices_j = sorted(range(len(j)), key=lambda i: j[i], reverse=True)
print("\nTop 10 sorted nodes by importance (PageRank) for Jacobi Method:")
for i in sorted_indices_j[:10]:
    print(f"Node {i} has PageRank score {j[i]}")

print("\n","-"*50)
# Sort and display results for the Gauss-Seidel
print(f"\nGauss-Seidel Method took {Gauss_end_time - Gauss_start} seconds")
sorted_indices_gs = sorted(range(len(gs)), key=lambda i: gs[i], reverse=True)
print("\nTop 10 sorted nodes by importance (PageRank) for Gauss-Seidel Method:")
for i in sorted_indices_gs[:10]:
    print(f"Node {i} has PageRank score {gs[i]}")

print("\n","="*50,"\n"," "*20, "Differences\n")

#Compute the differences between the methods
# Compute the differences between the methods using L2 norm
print("\nThe differences between each method with the L2 norm (Euclidean distance)")
diff_jp_l2 = np.linalg.norm(j - p, 2)  # L2 norm of difference
diff_gj_l2 = np.linalg.norm(j - gs, 2)  # L2 norm of difference
diff_gp_l2 = np.linalg.norm(gs - p, 2)  # L2 norm of difference

print(f"\nDifference between Jacobi and Power Method: {diff_jp_l2:.6f}")
print(f"Difference between Jacobi and Gauss-Seidel: {diff_gj_l2:.6f}")
print(f"Difference between Power Method and Gauss-Seidel: {diff_gp_l2:.6f}")






