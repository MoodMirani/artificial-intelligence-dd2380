import math
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
    for row in A:
        print(*row)

def forward_pass(transition_matrix_A, emissions_matrix_B, pi, emission_sequence):
    T = len(emission_sequence) # number of observations
    N = len(pi) # number of possible states

    alpha = [] 
    for i in range(T):
        alpha.append([]) 
    
    c = [] # array of scaling values at each timestep

    for i in range(N): # calculate alpha at step 1, with pi
        alpha[0].append(emissions_matrix_B[i][emission_sequence[0]]*pi[i])

    # normalize initial probabilities:
    c.append(1.0 / sum(alpha[0]))
    for i in range(N):
        alpha[0][i] *= c[0]

    for t in range(1, T):
        for x in range(N):
            emission = emission_sequence[t] # extract the next observation from the sequence
            emission_probability = emissions_matrix_B[x][emission] # find the probability of that observation at state x
            previous_transition_probability_sum = 0.0 
            for j in range(N):
                previous_transition_probability_sum += (transition_matrix_A[j][x] * alpha[t - 1][j]) # recurrence, we skip calculations already made

            alpha[t].append(emission_probability * previous_transition_probability_sum)
        # normalize alpha[t][i]
        c.append(1.0 / sum(alpha[t]))
        for i in range(N):
            alpha[t][i] *= c[t]

  # total probability of the whole sequence of observations at each t, scaling for each t
    return alpha, c


def backward_pass(transition_matrix_A, emissions_matrix_B, emission_sequence, c):
    # length of the emission sequence
    T = len(emission_sequence)
    # number of hidden states
    N = len(transition_matrix_A[0])

    # create an empty list for beta for each time step
    beta = [] 
    for i in range(T):
        beta.append([]) 

    # initialize the last row of beta with scaling factors
    last_row = []
    for i in range(N):
        last_row.append(c[T - 1])
    beta.append(last_row)

    # loop backwards over each time step
    for t in range(T - 2, -1, -1):
        # loop over each hidden state
        for i in range(N):
            # emission for the next time step
            emission = emission_sequence[t + 1]
            # sum the probabilities of transitioning to all possible next hidden states
            previous_transition_probability_sum = 0.0
            for j in range(N):
                previous_transition_probability_sum += (beta[t + 1][j] * transition_matrix_A[i][j] * emissions_matrix_B[j][emission])

            # append the result, scaled by the same factor as the corresponding c value, to beta
            beta[t].append(previous_transition_probability_sum * c[t])

    # return the list of backward probabilities for each hidden state and time step
    return beta

def compute_di_gamma_matrix(A, B, alpha, beta, observations):
    T = len(observations) # number of observations
    N = len(A[0]) # number of possible states

    di_gamma = [[[] for j in range(N)] for i in range(T)] # will eventually be T x N x N
    gamma = [[] for i in range(T - 1)] # will eventually be T x N

    for t in range(T - 1): # iterate over each observation except the last one
        for i in range(N): # iterate over each possible state
            gamma_sum = 0.0 # initialize gamma_sum for this i and t
            for j in range(N): # iterate over each possible state again
                # calculate the product of probabilities at this t and between states i and j
                # this will be one element of di_gamma[t][i] which will be appended to di_gamma later
                p1 = alpha[t][i]
                p2 = A[i][j]
                p3 = B[j][observations[t + 1]]
                p4 = beta[t + 1][j]
                product = p1 * p2 * p3 * p4 # (already scaled so we don't divide by sum(alpha[T - 1]))
                di_gamma[t][i].append(product) # add this product to di_gamma at time t, for state i and state j
                gamma_sum += product # add this product to the running sum for gamma[t] at state i

            gamma[t].append(gamma_sum) # add gamma_sum for this state i and time t

    gamma.append(alpha[T - 1][:]) # Special case for gamma at last index (see Stamp psuedocode)
    # gamma[T-1] is not calculated in the above loop, so we append it here separately

    return di_gamma, gamma
def estimate_lambda(di_gamma, gamma, emission_sequence, K):
    # K is number of possible observation types
    T = len(emission_sequence)
    N = len(di_gamma[0])

    pi_estimate = [gamma[0][:]] # estimate of initial state probabilities

    B_estimate = [] # estimate of emission matrix, eventually N x K
    for i in range(N):
        B_estimate.append([]) 

    A_estimate = [] # estimate of transition matrix, eventually N x N
    for i in range(N):
        A_estimate.append([]) 

    # estimate each element of A
    for i in range(N):
        denom = 0.0 # calculate the denominator, exclude last element at T from sum
        for t in range(T - 1):
            denom += gamma[t][i]

        for j in range(N):
            numer = 0 # calculate the numerator
            for t in range(T - 1):
                numer += di_gamma[t][i][j]
            A_estimate[i].append(numer / denom) # add the estimated value to the transition matrix A_estimate

    # estimate each element of B
    for i in range(N):
        denom = 0.0 # calculate the denominator
        for t in range(T):
            denom += gamma[t][i]

        for k in range(K):
            indicator_sum = 0.0 # calculate the numerator
            for t in range(T):
                if emission_sequence[t] == k:
                    indicator_sum += gamma[t][i]
            B_estimate[i].append(indicator_sum / denom) # add the estimated value to the emission matrix B_estimate

    return A_estimate, B_estimate, pi_estimate
def compute_log_prob(c):
    log_prob = 0
    for i in range(len(c)):
        #print(c[i])
        log_prob += math.log(c[i])

    return -log_prob

def baum_welch(est_A, est_B, est_pi, emission_sequence, K, max_iters=100):
    # K = number of observation types

    cur_iters = 0
    prev_log_prob = float("-inf")
    cur_log_prob = float("-inf")

    # Baum-Welch update loop
    while((prev_log_prob == float("-inf")) or
         ((cur_iters < max_iters) and (cur_log_prob - prev_log_prob > 0))):
         
        # Step 1: Compute the forward probabilities
        alpha, c = forward_algorithm(est_A, est_B, est_pi, emission_sequence)
        
        # Step 2: Compute the backward probabilities
        beta = backward_algorithm(est_A, est_B, emission_sequence, c)
        
        # Step 3: Calculate the di-gamma and gamma matrices
        di_gamma, gamma = calculate_di_gamma_matrices(est_A, est_B, alpha, beta, emission_sequence)
        
        # Step 4: Use the di-gamma and gamma matrices to estimate new values for the transition matrix, 
        # emission matrix, and initial state distribution
        est_A, est_B, est_pi = estimate_hmm_model(di_gamma, gamma, emission_sequence, K)
        
        # Step 5: Compute the log-likelihood of the observed emission sequence using the new estimates of the model parameters
        prev_log_prob = cur_log_prob
        cur_log_prob = compute_log_prob(c)
        
        # Step 6: Repeat until convergence or maximum number of iterations is reached
        cur_iters += 1

    return est_A, est_B, est_pi, cur_iters
        
  
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

    emission_types = len(emissions_matrix_B[0])
    est_A, est_B, est_pi = baum_welch(transition_matrix_A, emissions_matrix_B, pi, emission_sequence, emission_types, max_iters=100)
    print("Answer")
    print_matrix(est_pi)
    print("-------")
    print_matrix(est_A)
    print("-------")
    print_matrix(est_B)

main()