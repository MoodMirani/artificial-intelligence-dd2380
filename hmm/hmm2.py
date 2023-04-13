from sys import stdin

def make_matrix(row):
    elements = row.split(" ")
    rows = int(elements[0])
    columns = int(elements[1])
    matrix = []
    values = elements[2:]

    for r in range(rows):
        a_row = []
        for c in range(columns):
            a_row.append(float(values[r*columns+c]))
        matrix.append(a_row)
    return matrix

def viterbi_algorithm(transition_matrix_A, emissions_matrix_B, pi, emission_sequence):
    # Determine the number of observations and possible states.
    T = len(emission_sequence) # number of observations
    N = len(pi) # number of possible states

    # Initialize delta and delta_index matrices.
    delta = [] 
    for i in range(T):
        delta.append([]) 

    delta_index = []
    for i in range(T):
        delta_index.append([]) 

    # Calculate delta at step 1 with pi and emissions_matrix_B.
    for i in range(N):
        delta[0].append(emissions_matrix_B[i][emission_sequence[0]]*pi[i])

    # Compute delta and delta_index.
    for t in range(1, T):
        for x in range(N):
            # Find the probability of the current observation at state x.
            emission_probability = emissions_matrix_B[x][emission_sequence[t]]

            previous_emission_probabilities = []
            # Compute the probability of transitioning from each possible previous state to the current state x.
            for j in range(N):
                previous_emission_probabilities.append(transition_matrix_A[j][x] * delta[t - 1][j] * emission_probability)

            # Find the index and value of the highest probability in previous_emission_probabilities.
            max_prob = max(previous_emission_probabilities)
            max_index = previous_emission_probabilities.index(max_prob)

            # Append the index and value of the highest probability to delta_index and delta, respectively.
            delta_index[t].append(max_index)
            delta[t].append(max_prob)

    # Find the index of the highest probability at the last time step to start working backwards.
    state_sequence = [max(enumerate(delta[-1]), key=lambda x: x[1])[0]]

    # Use delta_idx to work backwards and determine the state sequence.
    for t in range(T - 2, -1, -1):
        state_sequence.insert(0, delta_index[t + 1][state_sequence[0]])
    return state_sequence
        
  
def main():
    # create matrices 
    transition_matrix_A = make_matrix(stdin.readline())
    emissions_matrix_B = make_matrix(stdin.readline())
    pi = make_matrix(stdin.readline())[0]
    emission_sequence = stdin.readline().split()

    # format the emission sequence array
    for i in range(len(emission_sequence)):
        emission_sequence[i] = (int(emission_sequence[i]))
    emission_sequence.pop(0)

    state_sequence = viterbi_algorithm(transition_matrix_A, emissions_matrix_B, pi, emission_sequence)
    state_sequence_string = ""
    for state in state_sequence:
        state_sequence_string = state_sequence_string + " " + str(state)
    
    print(state_sequence_string)

main()