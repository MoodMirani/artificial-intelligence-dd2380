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

def forward_algorithm(transition_matrix_A, emissions_matrix_B, pi, emission_sequence):
    T = len(emission_sequence) # number of observations
    N = len(pi) # number of possible states

    alpha = [] 
    for i in range(T):
        alpha.append([]) 
    
    for i in range(N): # calculate alpha at step 1, with pi
        alpha[0].append(emissions_matrix_B[i][emission_sequence[0]]*pi[i])

    for t in range(1, T):
        for x in range(N):
            emission = emission_sequence[t] # extract the next observation from the sequence
            emission_probability = emissions_matrix_B[x][emission] # find the probability of that observation at state x
            previous_transition_probability_sum = 0.0 
            for j in range(N):
                previous_transition_probability_sum += (transition_matrix_A[j][x] * alpha[t - 1][j]) # recurrence, we skip calculations already made

            alpha[t].append(emission_probability * previous_transition_probability_sum)
    return alpha
        
  
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

    print(sum(forward_algorithm(transition_matrix_A, emissions_matrix_B, pi, emission_sequence)[-1]))

main()