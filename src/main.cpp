#include <iostream>
#include "mt/tensor.hpp"

int main() {
    Tensor test({2, 3, 3});
    float val = 1.0f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                test.at({i, j, k}) = val++;
    test.print();
    return 0;
}
