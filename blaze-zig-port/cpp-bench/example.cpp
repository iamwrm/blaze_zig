// Simple example demonstrating Blaze C++ matrix operations

#include <iostream>
#include <iomanip>
#include <blaze/Blaze.h>

using namespace blaze;

int main() {
    std::cout << "=== Blaze C++ Example ===" << std::endl << std::endl;
    
    // Create a 3x3 matrix
    DynamicMatrix<double, rowMajor> A(3, 3);
    
    // Fill with values
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    
    std::cout << "Matrix A:" << std::endl;
    std::cout << A << std::endl;
    
    // Create diagonal matrix B
    DynamicMatrix<double, rowMajor> B(3, 3, 0.0);
    B(0, 0) = 1.0;
    B(1, 1) = 2.0;
    B(2, 2) = 3.0;
    
    std::cout << "Matrix B (diagonal):" << std::endl;
    std::cout << B << std::endl;
    
    // Matrix multiplication using MKL
    DynamicMatrix<double, rowMajor> C = A * B;
    
    std::cout << "C = A * B:" << std::endl;
    std::cout << C << std::endl;
    
    // Matrix addition
    DynamicMatrix<double, rowMajor> D = A + C;
    
    std::cout << "D = A + C:" << std::endl;
    std::cout << D << std::endl;
    
    // Scalar multiplication
    DynamicMatrix<double, rowMajor> E = 0.5 * A;
    
    std::cout << "E = 0.5 * A:" << std::endl;
    std::cout << E << std::endl;
    
    // Trace
    double trace_A = 0;
    double trace_C = 0;
    for (size_t i = 0; i < 3; ++i) {
        trace_A += A(i, i);
        trace_C += C(i, i);
    }
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Trace of A: " << trace_A << std::endl;
    std::cout << "Trace of C: " << trace_C << std::endl;
    
    std::cout << std::endl << "=== Example Complete ===" << std::endl;
    
    return 0;
}
