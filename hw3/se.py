# Sequential Estimator
from ugdg import gaussian_gen

if __name__ == '__main__':
    mu = 0.
    var_numerator = 0.
    for i in range(1, 200):
        v = gaussian_gen(3., 5.)
        print(f"Add data point : {v}")
        mu_updated = mu + (v - mu) / i
        var_numerator = var_numerator + (v-mu)*(v-mu_updated)
        var_updated = var_numerator / i

        mu, var = mu_updated, var_updated
        print(f"μ = {mu:<23} σ2 = {var:<23}")
