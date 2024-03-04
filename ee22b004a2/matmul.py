def matmul(A=[[]], B=[[]]):
    if A == [[]] or B == [[]]:
        raise TypeError("Two arguments are expected for matrix multiplication")

    # checking errors in A
    if type(A) in (list, range, tuple):
        for row in A:
            if type(row) in (list, range, tuple):
                for ele in row:
                    if type(ele) in (list, range, tuple):
                        raise TypeError("Matrices cannot have more than two indices")
                    else:
                        if not type(ele) in (int, float, complex):
                            raise TypeError("Only Numerical entries are permitted in matrices")
            else:
                raise TypeError("Given input A is a vector, not a matrix")
    else:
        raise TypeError("Given input A is not a matrix")

    # checking errors in B
    if type(B) in (list, range, tuple):
        for row in B:
            if type(row) in (list, range, tuple):
                for ele in row:
                    if type(ele) in (list, range, tuple):
                        raise TypeError("Matrices cannot have more than two indices")
                    else:
                        if not type(ele) in (int, float, complex):
                            raise TypeError("Only Numerical entries are permitted in matrices")
            else:
                raise TypeError("Given input B is a vector, not a matrix")
    else:
        raise TypeError("Given input B is not a matrix")

    # dimensions of the matrices
    n = len(A)
    m1 = len(A[0])
    m2 = len(B)
    k = len(B[0])

    # check if all the elements in a matrix are defined
    for i in range(n):  # iterate over all rows
        if len(A[i]) != m1:
            raise TypeError("Matrix A is ill-defined; row lengths are unequal")

    for i in range(m2):
        if len(B[i]) != k:
            raise TypeError("Matrix B is ill-defined; row lengths are unequal")

    if m1 != m2:
        raise ValueError("Matrix Multiplication is not compatible")

    C = [[0 for i in range(k)] for j in range(n)]  # empty product matrix

    for i in range(n):  # conducting the below process for different rows of C
        for j in range(m1):  # for a given row of C, updating all elements with the individual elementary products
            for K in range(k):  # generating a matrix product and adding it to an element of C
                C[i][K] += A[i][j] * B[j][K]

    return C