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

# create matrices
A_ = make_matrix(stdin.readline())
B_ = make_matrix(stdin.readline())
PI_ = make_matrix(stdin.readline())
OS_ = stdin.readline().split()
OS_n = []
for i in OS_:
    OS_n.append(int(i))
OS_n.pop(0)
print(OS_n)


alpha_1 = multiply_matrices(PI_[0], get_columns(B_, OS_n[0]))

def forward_algorithm(alpha, OS):
    if len(OS) == 0:
        print(round(sum(alpha), 6))
        return sum(alpha)
    sum_term = [sum(multiply_matrices(alpha, get_columns(A_, i))) for i in range(len(A_[0]))]
    current_alpha = multiply_matrices(sum_term, get_columns(B_, OS[0]))
    forward_algorithm(current_alpha, OS[1:])

forward_algorithm(alpha_1, OS_n[1:])
