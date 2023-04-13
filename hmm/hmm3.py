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

def print_matrix(A):
    An = len(A)
    Am = len(A[0])
    stdout = str(An) + " " + str(Am)
    for i in range(An):
        for j in range(Am):
            stdout += " " + str(A[i][j])
    print(stdout)

def forward_pass(A, B, pi, observations):
    T = len(observations)
    N = len(pi[0])
    alpha = [[] for i in range(T)] # Will eventually be T x N
    c = [] # array of scaling values at each timestep

    # Initializate first_obs probabilities for each hidden state
    first_obs = observations[0]
    for i in range(N):
        obs_init_prob = B[i][first_obs] * pi[0][i]
        alpha[0].append(obs_init_prob)

    # normalize initial probabilities:
    c.append(1.0 / sum(alpha[0]))
    for i in range(N):
        alpha[0][i] *= c[0]

    # compute alpha over all times
    for t in range(1, T):
        for i in range(N):
            cur_obs = observations[t]
            obs_prob = B[i][cur_obs]
            prev_obs_transition_sum = 0.0
            for j in range(N):
                prev_obs_transition_sum += (A[j][i] * alpha[t - 1][j])

            alpha[t].append(obs_prob * prev_obs_transition_sum)

        # normalize alpha[t][i]
        c.append(1.0 / sum(alpha[t]))
        for i in range(N):
            alpha[t][i] *= c[t]

  # total probability of the whole sequence of observations at each t, scaling for each t
    return alpha, c


def backward_pass(A, B, observations, c):
    T = len(observations)
    N = len(A[0])
    beta = [[] for i in range(T - 1)] # Will eventually be T x N
    last_beta_row = []
    for i in range(N):
        last_beta_row.append(c[T - 1])
    beta.append(last_beta_row)

    for t in range(T - 2, -1, -1):
        for i in range(N):
            cur_obs = observations[t + 1]
            prev_obs_transition_sum = 0.0
            for j in range(N):
                prev_obs_transition_sum += (beta[t + 1][j] * A[i][j] * B[j][cur_obs])

        beta[t].append(prev_obs_transition_sum * c[t]) # scale by same factor as alpha[t]

    # the probability of observing all future observations at each time step and hidden state
    return beta


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

def compute_di_gamma_matrix(A, B, alpha, beta, observations):
    T = len(observations)
    N = len(A[0])

    di_gamma = [[[] for j in range(N)] for i in range(T)] # will eventually be T x N x N
    gamma = [[] for i in range(T - 1)] # will eventually be T x N

    for t in range(T - 1):
        for i in range(N):
            gamma_sum = 0.0
            for j in range(N):
                p1 = alpha[t][i]
                p2 = A[i][j]
                p3 = B[j][observations[t + 1]]
                p4 = beta[t + 1][j]
                product = p1 * p2 * p3 * p4 # (already scaled so we don't divide by sum(alpha[T - 1]))
                di_gamma[t][i].append(product)
                gamma_sum += product

            gamma[t].append(gamma_sum)

    gamma.append(alpha[T - 1][:]) # Special case for gamma at last index (see Stamp psuedocode)
    return di_gamma, gamma

def estimate_lambda(di_gamma, gamma, observations, K):
    # K is number of possible observation types
    T = len(observations)
    N = len(di_gamma[0])

    A_est = [[] for i in range(N)] # eventually N x N
    B_est = [[] for i in range(N)] # eventually N x K
    pi_est = [gamma[0][:]]

    # estimate each element of A
    for i in range(N):
        denom = 0.0 # exclude last element at T from sum
        for t in range(T - 1):
            denom += gamma[t][i]

    for j in range(N):
        numer = 0
        for t in range(T - 1):
            numer += di_gamma[t][i][j]
        A_est[i].append(numer / denom)

    # estimate each element of B
    # TODO: should be using gamma sum that includes last element for B?
    for i in range(N):
        denom = 0.0
        for t in range(T):
            denom += gamma[t][i]

        for k in range(K):
            indicator_sum = 0.0
        for t in range(T):
            if observations[t] == k:
                indicator_sum += gamma[t][i]
        B_est[i].append(indicator_sum / denom)

  return A_est, B_est, pi_est


        
  
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