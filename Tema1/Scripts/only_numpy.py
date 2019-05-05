from string import ascii_letters, digits

import numpy


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


def start(file_path):
    matrix_coefficients, matrix_free_terms, matrix_unknown_terms, matrix_helper = read_from_file(file_path)
    if not matrix_coefficients:
        print("[x] Couldn't read and format information from {}".format(file_path))
        return

    for i in range(0, len(matrix_unknown_terms)):
        for j in range(0, len(matrix_helper)):
            if matrix_unknown_terms[i] not in matrix_helper[j]:
                matrix_coefficients[j].insert(i, 0)

    np_matrix_a = numpy.matrix(matrix_coefficients)
    np_matrix_b = numpy.matrix(matrix_free_terms)
    determinant = numpy.linalg.det(np_matrix_a)
    if not determinant:
        print("[x] Determinant is null")
        return

    inverse = numpy.linalg.inv(np_matrix_a)
    result = numpy.dot(inverse, np_matrix_b)

    print("Solution:")
    for i in range(0, len(matrix_unknown_terms)):
        print("{} = {}".format(matrix_unknown_terms[i], result.item(i)))


if __name__ == '__main__':
    start(r'..\Utils\equations.txt')
