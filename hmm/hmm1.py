from sys import stdin

# read input from file
#document = open("input0.in", "r")

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


def multiply_matrices(m1, m2):
    return [i * j for i, j in zip(m1, m2)]

def get_columns(matrix, col):
    return [row[col] for row in matrix]

def forward_algorithm():
    # we start by initialising alpha with the initial probability distribution
    alpha = multiply_matrices(pi[0], get_columns(emissions_matrix_B, emission_sequence[0]))
    print(alpha)
    emission_sequence_length = len(emission_sequence)
    for t in range(1, emission_sequence_length):
        test = 1
    return alpha

# create matrices 
transition_matrix_A = make_matrix(stdin.readline())
emissions_matrix_B = make_matrix(stdin.readline())
pi = make_matrix(stdin.readline())
emission_sequence = stdin.readline().split()
for i in range(len(emission_sequence)):
    emission_sequence[i] = (int(emission_sequence[i]))
emission_sequence.pop(0)


print("pi[0]: ", pi[0])
print("emissions[0]: ", pi[0])
print("emissions_matrix: ", emissions_matrix_B)
alpha = multiply_matrices(pi[0], get_columns(emissions_matrix_B, emission_sequence[0]))

forward_algorithm()
