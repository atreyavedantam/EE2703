# EE2703 - Applied Programming Lab
### Atreya Vedantam, EE22B004
## Assignment 1 - Matrix Multiplication

### Introduction

The objective of this code is to perform matrix multiplication between two matrices of compatible orders (the number of columns of the first matrix must equal the number of rows of the second matrix). It must check if the matrices are of proper sizes and if the elements in the matrices are of appropriate numeric datatypes.

### Approach

Matrix Multiplication between two matrices $A$ of order $n \times m_1$ and $B$ of order $m_2 \times k$ is defined only if $m_1 = m_2$. IF $A$ is a matrix of the form $$\begin{bmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\end{bmatrix}$$ and $B$ is a matrix of the form  $$\begin{bmatrix} b_{11} & b_{12}\\ b_{21} & b_{22} \\ b_{31} & b_{32}\end{bmatrix}$$ then their matrix product is given by  $$ C = \begin{bmatrix} a_{11}b_{11}+ a_{12}b_{21} + a_{13}b_{31} & a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32} \\ a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31} & a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32}\end{bmatrix}$$

## Algorithm

From the above paragraph we see that for the calculation of every matrix element in the product we need to consider a row of the matrix $A$ and a column of the matrix $B$ and sum over all their elements. We need to do this over all rows and columns of $C$, so this algorithm has a computational complexity $\mathbb{O}(n^3)$. 

### Procedure

We wish to update the elements of $C$ one product (of an element from $A$ and one from $B$) at a time. We do this updating row-wise. First we initialize the product $a_{11}b_{11}$ to $c_{11}$. Then we initialize $c_{12}$ as $a_{11}b_{12}$. This iteration is given by the innermost $\texttt{for}$ loop. Then we add the product $a_{12}b_{21}$ to $c_{11}$. Subsequently we add $a_{12}b_{22}$ to $c_{12}$. Similarly we update the third product to complete the creation of the elements $c_{11}$ and $c_{12}$. This loop is the middle $\texttt{for}$ loop. Finally we want to do this for all the rows of $C$, i.e. we want to update the elements $c_{21}$ and $c_{22}$ as well. This iteration is achieved by the outermost $\texttt{for}$ loop.

## Error Handling

All errors that are handled by this code are listed below.
- In the $\texttt{matmul}$ function, two entries are expected. If only one input or no inputs (is) are given, the appropriate error message is displayed.
- Both matrices must be nested iterables (lists/ ranges/ tuples) - exactly one iterable inside another. Note that [$\begin{matrix} 3 & 4 \\ \end{matrix}$] is considered a vector, not a matrix since it is not nested. One would instead have to enter [$\begin{matrix} [3] & [4]\\  \end{matrix}$]. If there is triple or more orders of nesting, an error is raised. If there is only a non-nested iterable or the input is of some other datatype, an error is raised.
- All elements in the matrices must belong to a numeric datatype - int, float or complex. An error is raised if otherwise.
- We also check that the number of elements in each row are equal (otherwise the matrix is ill-defined and matrix multiplication cannot be carried out).
- Finally the matrices must be compatible for multiplication: $m_1 = m_2$ as described above. This is also checked for.

## Impact of Error Handling on Computational Complexity

In the order of the various errors handled in the previous section, I specify below why it is feasible to conduct these checks.
- This is a simple check and does not consume appreciable time compared to the matrix multiplication.
- This check is an $\mathbb{O}(n^2)$ operation since it is checking the datatype of each element in each of the two matrices. If $n$ is large then this is also negligible compared to the actual matrix multiplication process which has a computational complexity $\mathbb{O}(n^3)$ (we are concerned with the complexity of the algorithm only when $n$ is large since only then will the time taken to run the program be appreciable).
- This again takes negligible time to perform.
- This check is an $\mathbb{O}(n)$ operation since we are parsing the outer level of nesting to compare the lengths of the rows. This order is also negligible compared to matrix multiplication.
- This takes negligible time too.

Overall, the complexity is not affected much by the error handling in the code and hence it is feasible to conduct such tests.

 <b>Justification for why it is <u>not</u> necessary to check for corner/ special cases in this assignment:</b> 
 - Matrix Multiplication does not involve any manner of division and hence we do not need to worry about Python's ability to handle division by zero. 
 - We also do not have any infinite recursive statements that may endanger the overall program. 

## Areas of Improvement

 - I have not optimized the RAM allocation in this program. We can use $\texttt{sys.getsizeof()}$ to identify areas of huge memory consumption (it turns out that lists hog a lot of memory). A more memory efficient method would be to use arrays from the $\texttt{array}$ module. However the downside is that it inhibits a lot of mathematics like complex number operations which I felt was necessary to have. Numpy would have been optimal in this case, however we were not allowed to use it for this particular assignment.
 - The state-of-the-art computational complexity for matrix multiplication is $\mathbb{O}(n^{2.37188})$ while my program is a bit more than $\mathbb{O}(n^{3})$. This is a possible area of improvement.

