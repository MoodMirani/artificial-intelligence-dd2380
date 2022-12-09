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

# create matrices
transition_matrix = make_matrix(stdin.readline())
emission_matrix = make_matrix(stdin.readline())
ispd = make_matrix(stdin.readline())

def multiply_matrices(m1, m2):
    # initialise new empty matrix
    matrix = []
    rows = len(m1)
    columns = len(m2[0])
    for r in range(rows):
        matrix.append([])
        for c in range(columns):
            matrix[r].append(0)
    
    # matrix multiplication
    for row in range(rows):
        for column in range(columns):
            for row_m2 in range(len(m2)):
                matrix[row][column] += m1[row][row_m2] * m2[row_m2][column]
    return matrix

# multiply the transition matrix with our current estimate of states
question2 = multiply_matrices(ispd, transition_matrix)

# result multiplied with the observation matrix.
question3 = multiply_matrices(question2, emission_matrix)

# output
output = str(len(question3)) + " " + str(len(question3[0]))

for row in question3:
        for column in row:
            output = output + " " + str(column)

print(output)



