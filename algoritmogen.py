import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_neural_network(weights, architecture):
    layers = []
    start = 0
    for i in range(len(architecture) - 1):
        end = start + architecture[i] * architecture[i+1]
        layer_weights = weights[start:end].reshape((architecture[i], architecture[i+1]))
        layers.append(layer_weights)
        start = end
    return layers

def forward_propagation(inputs, network):
    activation = inputs
    for layer in network:
        activation = sigmoid(np.dot(activation, layer))
    return activation

def xor_truth_table(network):
    print("\nTabla de verdad XOR:")
    print("X1 | X2 | Y_predicho")
    print("---+----+-----------")
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for input in inputs:
        output = forward_propagation(input, network)
        y_predicho = 1 if output[0] > 0.5 else 0
        print(f" {input[0]} |  {input[1]} |     {y_predicho}")
    return inputs, [forward_propagation(input, network)[0] for input in inputs]

def calculate_accuracy(y_true, y_pred):
    y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
    return np.mean(y_true == y_pred_binary) * 100

def original_fitness_function(weights):
    y = np.sum(weights)
    m = np.mean(weights)
    return (sigmoid(y + m) * sigmoid(-y - m)) * np.exp(-0.1 * abs(y))

def xor_fitness_function(weights, architecture):
    network = create_neural_network(weights, architecture)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([0, 1, 1, 0])
    outputs = [forward_propagation(input, network)[0] for input in inputs]
    accuracy = calculate_accuracy(y_true, outputs)
    return accuracy

def run_genetic_algorithm():
    # Parámetros del algoritmo genético
    POPULATION_SIZE = 50
    MUTATION_RATE = 0.1
    ARCHITECTURE = [2, 4, 1]  # Arquitectura de la red: 2 entradas, 4 neuronas ocultas, 1 salida
    NUM_WEIGHTS = sum(ARCHITECTURE[i] * ARCHITECTURE[i+1] for i in range(len(ARCHITECTURE) - 1))

    # Solicitar al usuario el número de generaciones
    while True:
        try:
            NUM_GENERATIONS = int(input("Por favor, ingrese el número de generaciones que desea ejecutar: "))
            if NUM_GENERATIONS > 0:
                break
            else:
                print("Por favor, ingrese un número positivo.")
        except ValueError:
            print("Por favor, ingrese un número entero válido.")

    # Inicialización de la población
    population = np.random.uniform(-1, 1, (POPULATION_SIZE, NUM_WEIGHTS))

    best_overall = None
    best_overall_original_fitness = float('-inf')
    best_overall_xor_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        # Evaluación de la aptitud
        original_fitness_scores = np.array([original_fitness_function(individual) for individual in population])
        xor_fitness_scores = np.array([xor_fitness_function(individual, ARCHITECTURE) for individual in population])
        
        # Selección de padres basada en la suma ponderada de ambas puntuaciones de aptitud
        combined_fitness = 0.5 * original_fitness_scores + 0.5 * xor_fitness_scores
        parents = population[np.argsort(combined_fitness)[-2:]]
        
        # Actualizar el mejor individuo global
        current_best_index = np.argmax(combined_fitness)
        current_best = population[current_best_index]
        current_best_original_fitness = original_fitness_scores[current_best_index]
        current_best_xor_fitness = xor_fitness_scores[current_best_index]
        
        if current_best_original_fitness > best_overall_original_fitness or current_best_xor_fitness > best_overall_xor_fitness:
            best_overall = current_best
            best_overall_original_fitness = current_best_original_fitness
            best_overall_xor_fitness = current_best_xor_fitness
        
        # Cruce (crossover)
        offspring = []
        for _ in range(POPULATION_SIZE - 2):
            parent1, parent2 = parents
            crossover_point = np.random.randint(1, NUM_WEIGHTS)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)
        
        offspring = np.array(offspring)
        
        # Mutación
        for i in range(len(offspring)):
            for j in range(NUM_WEIGHTS):
                if np.random.random() < MUTATION_RATE:
                    offspring[i, j] += np.random.normal(0, 0.1)
        
        # Nueva generación
        population = np.vstack((parents, offspring))
        
        # Calcular porcentajes
        theoretical_max = 0.25  # Ajustado para hacer el 100% desafiante pero alcanzable
        original_percentage = max(0, min(100, (current_best_original_fitness / theoretical_max) * 100))
        xor_percentage = current_best_xor_fitness
        
        # Imprimir el mejor individuo de esta generación
        print(f"Generación {generation + 1}:")
        print(f"  Aptitud original = {original_percentage:.2f}%")
        print(f"  Precisión XOR = {xor_percentage:.2f}%")

    print(f"\nMejor solución encontrada después de {NUM_GENERATIONS} generaciones:")
    best_network = create_neural_network(best_overall, ARCHITECTURE)
    inputs, outputs = xor_truth_table(best_network)
    y_true = np.array([0, 1, 1, 0])
    accuracy = calculate_accuracy(y_true, outputs)
    print(f"Precisión XOR final: {accuracy:.2f}%")
    
    final_original_percentage = max(0, min(100, (best_overall_original_fitness / theoretical_max) * 100))
    print(f"Aptitud original final: {final_original_percentage:.2f}%")

# Bucle principal para ejecutar el programa múltiples veces
while True:
    run_genetic_algorithm()
    
    # Preguntar al usuario si desea ejecutar el programa nuevamente
    respuesta = input("\n¿Desea ejecutar el programa nuevamente? (s/n): ").lower()
    if respuesta != 's':
        print("Gracias por usar el programa. ¡Hasta luego!")
        break