//
// Created by 施奕成 on 2023/3/5.
//

#pragma once
#include <random>
#include <vector>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <algorithm>

class Solver {
public:
    explicit Solver() : n_(-1), lambda_(-1) {};
    std::vector<double> solve(int, double, const std::string &);
private:
    int n_;
    double lambda_;
    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> ata_reg;
    std::vector<double> b;
    void read_Ab_mat();
    static void lu_decompose(std::vector<std::vector<double>> &, std::vector<int> &);
    static void lu_substitute(std::vector<std::vector<double>> &,
                              std::vector<int> &,
                              std::vector<double> &);
    void lu_inv_ata_reg(std::vector<std::vector<double>> &);
    void a_matmul_b(std::vector<double> &);
    static void transpose(std::vector<std::vector<double>>& mat);
};
