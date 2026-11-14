import random

# Network topology (Graph)
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'D': 3, 'E': 8},
    'C': {'A': 2, 'D': 5, 'E': 6},
    'D': {'B': 3, 'C': 5, 'E': 3},
    'E': {'B': 8, 'C': 6, 'D': 3}
}

# Input
source = input("Enter source router: ").upper()
destination = input("Enter destination router: ").upper()

ants = 10
iterations = 50
alpha = 1
beta = 3
rho = 0.5
Q = 100

# Initialize pheromone levels
pheromone = {}

for node in graph:
    for neighbor in graph[node]:
        pheromone[(node, neighbor)] = 1.0


best_path = None
best_cost = float('inf')

# ACO Algorithm
for iteration in range(iterations):

    all_paths = []

    for ant in range(ants):

        current = source
        path = [current]
        cost = 0

        while current != destination:

            neighbors = [
                n for n in graph[current]
                if n not in path
            ]

            if not neighbors:
                break

            # Calculate probabilities
            probabilities = []

            for n in neighbors:
                tau = pheromone[(current, n)]
                eta = 1 / graph[current][n]

                probabilities.append(
                    (tau ** alpha) * (eta ** beta)
                )

            total = sum(probabilities)

            probabilities = [
                p / total for p in probabilities
            ]

            # Select next router
            next_node = random.choices(
                neighbors, weights=probabilities
            )[0]

            cost += graph[current][next_node]
            path.append(next_node)
            current = next_node

        # Store successful paths
        if current == destination:

            all_paths.append((path, cost))

            if cost < best_cost:
                best_cost = cost
                best_path = path

    # Pheromone evaporation
    for edge in pheromone:
        pheromone[edge] *= (1 - rho)

    # Pheromone update
    for path, cost in all_paths:

        deposit = Q / cost

        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            pheromone[edge] += deposit

# Output
print("\n----- ACO NETWORK ROUTING RESULT -----")

if best_path:
    print("Source Router:", source)
    print("Destination Router:", destination)
    print("Optimal Path:", " -> ".join(best_path))
    print("Minimum Network Cost:", best_cost)
else:
    print("No path found between the routers.")
