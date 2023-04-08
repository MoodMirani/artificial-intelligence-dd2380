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

def forward_algorithm(alpha, emissions):
    if len(emissions) == 0:
        print(round(sum(alpha), 6))
        return sum(alpha)
    sumList = [sum(multiply_matrices(alpha, get_columns(transition_matrix, i))) for i in range(len(transition_matrix[0]))]
    current_alpha = multiply_matrices(sumList, get_columns(emission_matrix, emissions[0]))
    forward_algorithm(current_alpha, emissions[1:])

# create matrices
transition_matrix = make_matrix(stdin.readline())
emission_matrix = make_matrix(stdin.readline())
pi = make_matrix(stdin.readline())
emissions = stdin.readline().split()
emission_n = []
for i in emissions:
    emission_n.append(int(i))
emission_n.pop(0)

alpha_1 = multiply_matrices(pi[0], get_columns(emission_matrix, emission_n[0]))

forward_algorithm(alpha_1, emission_n[1:])
