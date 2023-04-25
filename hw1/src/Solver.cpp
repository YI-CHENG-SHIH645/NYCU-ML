//
// Created by 施奕成 on 2023/3/5.
//
#include "Solver.h"
using namespace std;

void Solver::transpose(vector<vector<double>> &mat) {
  int rows = (int)mat.size();
  int cols = (int)mat[0].size();
  for (int i=0; i<rows-1; ++i) {
    for (int j=i+1; j<cols; ++j) {
      mat[j][i] = mat[i][j];
    }
  }
}

void Solver::read_Ab_mat() {
  if(!A.empty()) {
    for(auto & row : A) {
      vector<double> new_row(n_);
      for (int i=n_-1; i>0; --i)
        new_row.at(n_-1-i) = pow(row[row.size()-2], i);
      new_row.at(n_-1) = 1.;
      row = new_row;
    }
  }
  else {
    string line, str_num;
    while (getline(cin, line)) {
      if (line.empty()) break;
      vector<double> A_row;
      stringstream ss(line);
      while (ss.good()) {
        getline(ss, str_num, ',');
        if (!ss.good()) b.push_back(stod(str_num));
        else {
          for (int i=n_-1; i>0; --i)
            A_row.push_back(pow(stod(str_num), i));
        }
      }
      A_row.push_back(1.);
      A.push_back(A_row);
    }
  }
}

void Solver::lu_decompose(vector<vector<double>> & lu,
                          vector<int> & permute) {
  int n = (int)lu.size();
  vector<double> norm_scale(n, 0.);
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) { norm_scale[i] = max(norm_scale[i], fabs(lu[i][j])); }
    norm_scale[i] = 1./norm_scale[i];
  }

  for(int j=0; j<n; j++) {
    // cal U : a_01(copy), a_02(copy), a_12(a_12 - a_10 * a_02)
    for(int i=0; i<j; i++) {
      for(int k=0; k<i; k++) {
        lu[i][j] -= lu[i][k] * lu[k][j];
      }
    }

    double abs_max = 0., tmp;
    int max_row;
    // cal L : a_00(copy), a_10(copy), a_20(copy),
    //         a_11(a_11-a_10*a_01), a_21(a_21-a_20*a_01),
    //         a_22(a_22-a_20*a_02-a_21*a_12)
    for (int i=j; i<n; i++) {
      for (int k=0; k<j; k++) {
        lu[i][j] -= lu[i][k] * lu[k][j];
      }
      tmp = fabs(lu[i][j]) * norm_scale[i];
      if (tmp >= abs_max) {
        abs_max = tmp;
        max_row = i;
      }
    }

    // partial pivoting
    if (max_row != j) {
      if ((j == (n-2)) && (lu[j][j+1] == 0.)) {
        max_row = j;
      }
      else {
        for (int k=0; k<n; k++)
          std::swap(lu[j][k], lu[max_row][k]);
        norm_scale[max_row] = norm_scale[j];
      }
    }

    permute[j] = max_row;
    if(lu[j][j] == 0.) {
      lu[j][j] = 1e-30;
    }
    if(j != (n-1))
      for(int i = (j+1) ; i < n; i++)
        lu[i][j] /= lu[j][j];
  }
}

void Solver::lu_substitute(std::vector<std::vector<double>> & lu,
        std::vector<int> & permute, std::vector<double> & ans) {

  int n = (int)lu.size();

  for(int i=0; i<n; i++) {

    double tmp = ans[permute[i]];
    ans[permute[i]] = ans[i];

    // forward substitution
    for(int j=(i-1); j>=0; j--) {
      tmp -= lu[i][j] * ans[j];
    }
    ans[i] = tmp;
  }

  // backward substitution
  for(int i=(n-1); i>=0; i--) {
    for(int j = (i+1); j < n ; j++ ) {
      ans[i] -= lu[i][j] * ans[j];
    }
    ans[i] /= lu[i][i] ;
  }
}

void Solver::lu_inv_ata_reg(vector<vector<double>> & ata_reg_inv) {
  ata_reg.clear();
  int c_sz = (int)A[0].size();
  int r_sz = (int)A.size();
  for(int c=0; c<c_sz; ++c) {
    double v;
    vector<double> row;
    for(int c2=0; c2<c_sz; ++c2) {
      v = 0;
      for(int r=0; r<r_sz; ++r) {
        v += A[r][c] * A[r][c2];
        if(c == c2 && c != c_sz-1) v += lambda_;
      }
      row.push_back(v);
    }
    ata_reg.push_back(row);
  }

  auto lu(ata_reg);
  vector<int> permute(lu.size());
  lu_decompose(lu, permute);

  for(int j=0; j<c_sz; ++j) {
    vector<double> ans(c_sz, 0.);
    ans[j] = 1.;
    lu_substitute(lu, permute, ans);
    ata_reg_inv.push_back(ans);
  }
  transpose(ata_reg_inv);
}

void Solver::a_matmul_b(vector<double> & atb) {
  for(int c=0; c<(int)A[0].size(); ++c) {
    double v = 0;
    for(int r=0; r<(int)A.size(); ++r) {
      v += A[r][c] * b[r];
    }
    atb.push_back(v);
  }
}

vector<double> Solver::solve(int n, double lambda, const std::string & method) {
  n_ = n; lambda_ = lambda;
  read_Ab_mat();
  vector<double> atb, best_params;
  a_matmul_b(atb);

  if(method == "LU") {
    vector<vector<double>> ata_reg_inv;
    lu_inv_ata_reg(ata_reg_inv);

    for (int r=0; r<(int)ata_reg_inv.size(); ++r) {
      double v = 0;
      for (int c=0; c<(int)ata_reg_inv[0].size(); ++c) {
        v += ata_reg_inv[r][c] * atb[c];
      }
      best_params.push_back(v);
    }
  }
  else if(method == "newton") {
    vector<vector<double>> inv_hessian;
    lu_inv_ata_reg(inv_hessian);

    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_real_distribution<double> dist {-1., 1.};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    vector<double> x_n(ata_reg.size(), 0);
    generate(begin(x_n), end(x_n), gen);
    vector<double> x_n1(x_n);

    while(true) {
      vector<double> grad_vec(ata_reg.size(), 0);
      for(int i=0; i<(int)ata_reg.size(); ++i) {
        for(int j=0; j<(int)ata_reg.size(); ++j) {
          grad_vec[i] += 2 * ata_reg[i][j] * x_n[j];
        }
        grad_vec[i] -= 2 * atb[i];
      }

      for(int i=0; i<(int)inv_hessian.size(); ++i) {
        for(int j=0; j<(int)inv_hessian.size(); ++j) {
          x_n1[i] -= 0.5 * inv_hessian[i][j] * grad_vec[j];
        }
      }
      double iter_max_diff = 1e-30;
      for(int i=0; i<(int)x_n1.size(); ++i)
        iter_max_diff = max(iter_max_diff, fabs(x_n1[i] - x_n[i]));
      cout << "iter_max_diff: " << iter_max_diff << endl;
      if(iter_max_diff < 1e-3) break;
      x_n = x_n1;
    }
    best_params = x_n1;
  }
  return best_params;
}
