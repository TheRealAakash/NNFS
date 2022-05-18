//
// Created by Aakash Kumar on 5/17/22.
//
#include "iostream"
#include <vector>
#include <random>


#include "Layers.h"

struct Matrix {
    int rows;
    int cols;
    std::vector<double> data;

    void create(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        data.resize(rows * cols);
    }

    // Matrix setter
    void set(int row, int col, double value) {
        data[row * cols + col] = value;
    }

    // Matrix getter
    double get(int row, int col) {
        return data[row * cols + col];
    }


};

int main() {
    std::cout << "test";
}