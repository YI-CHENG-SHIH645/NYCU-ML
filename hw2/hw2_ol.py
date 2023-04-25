import numpy as np
from scipy.special import factorial

if __name__ == '__main__':
    with open("testfile.txt", "r") as f:
        data = f.readlines()
    data = [np.array(list(d[:-1])).astype(np.int8) for d in data]
    a = 10
    b = 1
    for i, d in enumerate(data):
        print(f"Case {i+1}: {''.join(list(d.astype(str)))}")
        N = len(d)
        m = sum(d)
        print(f"N={N}, m={m}")
        mle = factorial(N)/(factorial(m)*factorial(N-m)) * (m/N)**m * (1-m/N)**(N-m)
        print(f"Likelihood: {mle:.4f} with p={m/N:.4f} in this case")
        # beta_prior = np.random.beta(a, b, 1)
        print(f"Beta prior:     a={a}, b={b}")
        a = m+a
        b = N-m+b
        # beta_posterior = np.random.beta(a, b, 1)[0]
        print(f"Beta posterior: a={a}, b={b}")
        print(f"p(θ_MAP) = {a/(a+b):.4f}")
        print(f"p(variance of θ) = {a*b/((a+b)**2*(a+b+1)):.4f}")
        print()
