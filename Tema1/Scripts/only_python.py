from string import ascii_letters, digits
from copy import deepcopy


__author__ = "Mihaila Alexandra Ioana"
__version__ = "1.1"
__status__ = "Dev"


def read_from_file(file_path):
    matrix_free_terms = list()
    matrix_coefficients = list()
    matrix_unknown_terms = list()
    matrix_helper = list()

    try:
        with open(file_path, 'r') as fin:
            line = fin.readline()
            while line:
                if not line.strip():
                    line = fin.readline()
                    continue

                current_coefficients = list()
                current_helper = list()
                free_term = line.split('=')[-1].strip()
                matrix_free_terms.append([int(free_term)])
                left_part = line.split('=')[0].strip()
                sign, coefficient, unknown_term = 1, None, None

                for section in left_part.split():
                    section = section.strip()
                    if section == '-':
                        sign = -1
                    elif section == '+':
                        continue
                    else:
                        coefficient = section.strip(ascii_letters)
                        if not coefficient:
                            coefficient = 1 * sign
                        else:
                            coefficient = int(coefficient) * sign

                        unknown_term = section.strip(digits)
                        current_helper.append(unknown_term)
                        if unknown_term not in matrix_unknown_terms:
                            matrix_unknown_terms.append(unknown_term)

                        current_coefficients.append(coefficient)
                        sign, coefficient, unknown_term = 1, None, None

                matrix_coefficients.append(current_coefficients)
                matrix_helper.append(current_helper)
                line = fin.readline()
    except Exception as error:
        print(error)
        return None, None

    return matrix_coefficients, matrix_free_terms, matrix_unknown_terms, matrix_helper


def calc_determinant(matrix):
    # term_list = list()
    if len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    d1 = (matrix[0][0] * matrix[1][1] * matrix[2][2] +
          matrix[0][1] * matrix[1][2] * matrix[2][0] +
          matrix[0][2] * matrix[1][0] * matrix[2][1])

    d2 = (matrix[0][2] * matrix[1][1] * matrix[2][0] +
          matrix[1][0] * matrix[0][1] * matrix[2][2] +
          matrix[0][0] * matrix[2][1] * matrix[1][2])

    return d1 - d2


def calc_transpose(matrix):
    return [[matrix[j][i] for j in range(0, len(matrix))] for i in range(0, len(matrix[0]))]


def calc_adjoint(matrix):
    new_matrix = list()
    for row in range(0, len(matrix)):
        new_matrix.append(list())

    for row in range(0, len(matrix)):
        for column in range(0, len(matrix[row])):
            copied_matrix = deepcopy(matrix)
            copied_matrix = copied_matrix[0:row] + copied_matrix[(row + 1):]
            for i in range(0, len(copied_matrix)):
                del copied_matrix[i][column]

            el = (-1) ** (row + column + 2) * calc_determinant(copied_matrix)
            new_matrix[row].append(el)

    return new_matrix


def calc_inverse(matrix, det):
    new_matrix = list()
    for row in range(0, len(matrix)):
        new_matrix.append(list())

    for row in range(0, len(matrix)):
        for column in range(0, len(matrix[row])):
            new_matrix[row].append(matrix[row][column] / det)

    return new_matrix


def calc_dot(matrix_a, matrix_b):
    new_matrix = list()
    for i in range(0, len(matrix_a)):
        new_matrix.append([0] * len(matrix_b[0]))

    for i in range(0, len(matrix_a)):
        for j in range(0, len(matrix_b[0])):
            for k in range(0, len(matrix_b)):
                new_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return new_matrix


def start(file_path):
    matrix_coefficients, matrix_free_terms, matrix_unknown_terms, matrix_helper = read_from_file(file_path)
    if not matrix_coefficients:
        print("[x] Couldn't read and format information from {}".format(file_path))
        return

    for i in range(0, len(matrix_unknown_terms)):
        for j in range(0, len(matrix_helper)):
            if matrix_unknown_terms[i] not in matrix_helper[j]:
                matrix_coefficients[j].insert(i, 0)

    determinant = calc_determinant(matrix_coefficients)
    if not determinant:
        print("[x] Determinant is null")
        return

    transpose = calc_transpose(matrix_coefficients)
    adjoint = calc_adjoint(transpose)
    inverse = calc_inverse(adjoint, determinant)
    dot_product = calc_dot(inverse, matrix_free_terms)

    print("Solution:")
    for i in range(0, len(matrix_unknown_terms)):
        print("{} = {}".format(matrix_unknown_terms[i], dot_product[i][0]))


if __name__ == '__main__':
    start(r'..\Utils\equations.txt')
