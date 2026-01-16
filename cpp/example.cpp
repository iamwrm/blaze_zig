#include <blaze/Blaze.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Blaze C++ Matrix Operations Example\n";
    std::cout << "====================================\n\n";

    // Create some matrices
    blaze::DynamicMatrix<double> A{{1.0, 2.0, 3.0},
                                    {4.0, 5.0, 6.0}};

    blaze::DynamicMatrix<double> B{{7.0,  8.0},
                                    {9.0,  10.0},
                                    {11.0, 12.0}};

    // Matrix multiplication
    blaze::DynamicMatrix<double> C = A * B;

    std::cout << "Matrix A (2x3):\n";
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.columns(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << A(i, j);
        }
        std::cout << '\n';
    }

    std::cout << "\nMatrix B (3x2):\n";
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.columns(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << B(i, j);
        }
        std::cout << '\n';
    }

    std::cout << "\nC = A * B (2x2):\n";
    for (size_t i = 0; i < C.rows(); ++i) {
        for (size_t j = 0; j < C.columns(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << C(i, j);
        }
        std::cout << '\n';
    }

    // Vector operations
    blaze::DynamicVector<double> v1{1.0, 2.0, 3.0};
    blaze::DynamicVector<double> v2{4.0, 5.0, 6.0};

    double dot_product = blaze::inner(v1, v2);
    std::cout << "\nVector v1: [" << v1[0] << ", " << v1[1] << ", " << v1[2] << "]\n";
    std::cout << "Vector v2: [" << v2[0] << ", " << v2[1] << ", " << v2[2] << "]\n";
    std::cout << "Dot product v1 . v2 = " << dot_product << "\n";

    // Element-wise operations
    blaze::DynamicVector<double> v3 = v1 + v2;
    std::cout << "v1 + v2 = [" << v3[0] << ", " << v3[1] << ", " << v3[2] << "]\n";

    blaze::DynamicVector<double> v4 = v1 * 2.0;
    std::cout << "v1 * 2 = [" << v4[0] << ", " << v4[1] << ", " << v4[2] << "]\n";

    std::cout << "\nBlaze example completed successfully!\n";
    return 0;
}
