import numpy as np

# ==========================================
# 1. SIMULATION SETUP & PROBLEM DATA
# ==========================================
HOURS = 24
POP_SIZE = 100
GENERATIONS = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITE_COUNT = 2

# Appliance Data: [Power (kW), Required Runtime Hours]
APPLIANCES = {
    "Air Conditioner": {"power": 2.0, "runtime": 6},
    "Washing Machine": {"power": 1.2, "runtime": 2},
    "Water Heater":    {"power": 3.0, "runtime": 3},
    "Refrigerator":    {"power": 0.3, "runtime": 24} # Baseload (always on)
}
APP_NAMES = list(APPLIANCES.keys())
N_APPS = len(APP_NAMES)

# Grid Constraints
MAX_GRID_LIMIT = 5.0  # Max kW household can draw per hour safely

# Time-of-Use (ToU) Tariff Profile (Price per kWh based on hour of the day)
# Off-Peak: 00:00-06:00 (Rs. 4), Shoulder: 06:00-18:00 & 22:00-24:00 (Rs. 7), Peak: 18:00-22:00 (Rs. 12)
TARIFF = np.array([4]*6 + [7]*12 + [12]*4 + [7]*2)

# ==========================================
# 2. GENETIC ALGORITHM CORE FUNCTIONS
# ==========================================

def initialize_population():
    """Generates random initial chromosome matrices (Size: N_APPS x HOURS)"""
    population = []
    for _ in range(POP_SIZE):
        # Generate a random binary matrix
        chromosome = np.random.randint(2, size=(N_APPS, HOURS))

        # Hard constraint enforcement: Refrigerator must run 24/7
        ref_idx = APP_NAMES.index("Refrigerator")
        chromosome[ref_idx, :] = 1

        population.append(chromosome)
    return population

def compute_fitness(chromosome):
    """Calculates total cost + penalty functions. Maximises 1 / (Cost + Penalties)"""
    total_cost = 0.0
    runtime_penalty = 0.0
    grid_penalty = 0.0

    # Calculate hourly load vector
    hourly_load = np.zeros(HOURS)

    for idx, name in enumerate(APP_NAMES):
        power = APPLIANCES[name]["power"]
        required_run = APPLIANCES[name]["runtime"]

        # Actual operational hours assigned by chromosome
        actual_run = np.sum(chromosome[idx, :])

        # 1. Runtime Constraint Penalty (Heavy penalty if device doesn't finish cycle)
        if actual_run != required_run:
            runtime_penalty += abs(actual_run - required_run) * 500  # Penalty scale factor

        # Accumulate power drawn at each hour
        hourly_load += chromosome[idx, :] * power

    # 2. Electricity Bill Calculation
    total_cost = np.sum(hourly_load * TARIFF)

    # 3. Peak Grid Demand Capacity Penalty
    for t in range(HOURS):
        if hourly_load[t] > MAX_GRID_LIMIT:
            grid_penalty += (hourly_load[t] - MAX_GRID_LIMIT) * 1000  # Heavy penalty for tripping breaker

    # Total Objective Score (To minimize)
    total_score = total_cost + runtime_penalty + grid_penalty

    # Return fitness score (inverse function for GA maximization goal)
    return 1.0 / (total_score + 1e-6)

def selection(population, fitnesses):
    """Tournament selection mechanism"""
    tournament_size = 3
    selected = []
    for _ in range(POP_SIZE):
        indices = np.random.choice(POP_SIZE, tournament_size, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        selected.append(population[best_idx].copy())
    return selected

def crossover(parent1, parent2):
    """Single-point crossover applied across the time dimension"""
    if np.random.rand() < CROSSOVER_RATE:
        cp = np.random.randint(1, HOURS) # Choose crossover point hour
        child1 = np.hstack((parent1[:, :cp], parent2[:, cp:]))
        child2 = np.hstack((parent2[:, :cp], parent1[:, cp:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(chromosome):
    """Bit-flip mutation strategy preserving the continuous baseload"""
    for idx in range(N_APPS):
        if APP_NAMES[idx] == "Refrigerator":
            continue # Skip mutating continuous baseload profile

        for t in range(HOURS):
            if np.random.rand() < MUTATION_RATE:
                chromosome[idx, t] = 1 - chromosome[idx, t] # Flip 0 to 1 or 1 to 0
    return chromosome

# ==========================================
# 3. MAIN GA RUNTIME EXECUTIVE
# ==========================================

if __name__ == "__main__":
    # 1. Initialize
    population = initialize_population()

    print(f"Starting Bio-Inspired Energy Optimizer (Generations: {GENERATIONS})...\n")

    for gen in range(1, GENERATIONS + 1):
        # Compute fitness vector
        fitnesses = [compute_fitness(ind) for ind in population]

        # Handle Elitism
        sorted_indices = np.argsort(fitnesses)[::-1]
        next_gen = [population[i].copy() for i in sorted_indices[:ELITE_COUNT]]

        # Selection
        selected_parents = selection(population, fitnesses)

        # Breed next generation
        while len(next_gen) < POP_SIZE:
            p1_idx = np.random.randint(POP_SIZE)
            p2_idx = np.random.randint(POP_SIZE)
            c1, c2 = crossover(selected_parents[p1_idx], selected_parents[p2_idx])
            next_gen.append(mutate(c1))
            if len(next_gen) < POP_SIZE:
                next_gen.append(mutate(c2))

        population = next_gen

        # Track historical execution progress every 50 loops
        if gen == 1 or gen % 50 == 0:
            best_fit = max(fitnesses)
            best_cost = (1.0 / best_fit)
            print(f"Generation {gen:03d} -> Current Minimized Operating Cost Metric value: Rs. {best_cost:.2f}")

    # ==========================================
    # 4. FINAL OUTPUT DECODING & REPORTING
    # ==========================================
    final_fitnesses = [compute_fitness(ind) for ind in population]
    best_chromosome = population[np.argmax(final_fitnesses)]

    print("\n" + "="*50)
    print(" OPTIMIZATION TASK COMPLETED SUCCESFULLY")
    print("="*50)

    final_hourly_load = np.zeros(HOURS)
    for idx, name in enumerate(APP_NAMES):
        active_hours = np.where(best_chromosome[idx, :] == 1)[0]
        final_hourly_load += best_chromosome[idx, :] * APPLIANCES[name]["power"]
        print(f"{name:<16} scheduled hours: {list(active_hours)}")

    final_bill = np.sum(final_hourly_load * TARIFF)
    print("-"*50)
    print(f"Optimal Standard Energy Bill: Rs. {final_bill:.2f} / day")
    print(f"Peak Power Drawn: {np.max(final_hourly_load):.2f} kW (Limit Check: {np.max(final_hourly_load) <= MAX_GRID_LIMIT})")
    print("="*50)

     
Starting Bio-Inspired Energy Optimizer (Generations: 200)...

Generation 001 -> Current Minimized Operating Cost Metric value: Rs. 7509.80
Generation 050 -> Current Minimized Operating Cost Metric value: Rs. 203.20
Generation 100 -> Current Minimized Operating Cost Metric value: Rs. 194.20
Generation 150 -> Current Minimized Operating Cost Metric value: Rs. 185.20
Generation 200 -> Current Minimized Operating Cost Metric value: Rs. 179.20

==================================================
 OPTIMIZATION TASK COMPLETED SUCCESFULLY
==================================================
Air Conditioner  scheduled hours: [np.int64(0), np.int64(1), np.int64(2), np.int64(4), np.int64(14), np.int64(19)]
Washing Machine  scheduled hours: [np.int64(5), np.int64(23)]
Water Heater     scheduled hours: [np.int64(3), np.int64(5), np.int64(23)]
Refrigerator     scheduled hours: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9), np.int64(10), np.int64(11), np.int64(12), np.int64(13), np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22), np.int64(23)]
--------------------------------------------------
Optimal Standard Energy Bill: Rs. 179.20 / day
Peak Power Drawn: 4.50 kW (Limit Check: True)
==================================================
