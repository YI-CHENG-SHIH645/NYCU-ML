from hw1 import Solver
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def plot(model, title, pic_file_name):
    plt.clf()
    y_reg = model(x_reg)
    plt.scatter(x, y)
    plt.plot(x_reg, y_reg)
    plt.title(title)
    plt.savefig(pic_file_name)


if __name__ == '__main__':
    xy = np.loadtxt("test.txt", delimiter=',')
    x, y = xy[:, 0], xy[:, 1]
    x_reg = np.linspace(-6, 6, 100)

    solver = Solver()
    LU_n2_lam0 = np.poly1d(solver.solve(2, 0, "LU"))
    newton_n2_lam0 = np.poly1d(solver.solve(2, 0, "newton"))
    print(f"LSE(n=2, lambda=0):\nFitting line:{LU_n2_lam0}\n"
          f"Total error: {len(x)*mean_squared_error(y, LU_n2_lam0(x)):.4f}\n")
    plot(LU_n2_lam0, "LSE(n=2, lambda=0)", "LSE_n2_lam0.png")
    print(f"Newton's Method(n=2, lambda=0):\nFitting line:{newton_n2_lam0}\n"
          f"Total error: {len(x)*mean_squared_error(y, newton_n2_lam0(x)):.4f} \n\n")
    plot(newton_n2_lam0, "Newton's Method(n=2, lambda=0)", "Newton_n2_lam0.png")

    LU_n3_lam0 = np.poly1d(solver.solve(3, 0, "LU"))
    newton_n3_lam0 = np.poly1d(solver.solve(3, 0, "newton"))
    print(f"LSE(n=3, lambda=0):\nFitting line:{LU_n3_lam0}\n"
          f"Total error: {len(x)*mean_squared_error(y, LU_n3_lam0(x)):.4f}\n")
    plot(LU_n3_lam0, "LSE(n=3, lambda=0)", "LSE_n3_lam0.png")
    print(f"Newton's Method(n=3, lambda=0):\nFitting line:{newton_n3_lam0}\n"
          f"Total error: {len(x)*mean_squared_error(y, newton_n3_lam0(x)):.4f}\n\n")
    plot(newton_n3_lam0, "Newton's Method(n=3, lambda=0)", "Newton_n3_lam0.png")

    LU_n3_lam1000 = np.poly1d(solver.solve(3, 10000, "LU"))
    newton_n3_lam10000 = np.poly1d(solver.solve(3, 10000, "newton"))
    print(f"LSE(n=3, lambda=10000):\nFitting line:{LU_n3_lam1000}\n"
          f"Total error: {len(x)*mean_squared_error(y, LU_n3_lam1000(x)):.4f}\n")
    plot(LU_n3_lam1000, "LSE(n=3, lambda=10000)", "LSE_n3_lam10000.png")
    print(f"Newton's Method(n=3, lambda=10000):\nFitting line:{newton_n3_lam10000}\n"
          f"Total error: {len(x)*mean_squared_error(y, newton_n3_lam10000(x)):.4f}\n")
    plot(newton_n3_lam10000, "Newton's Method(n=3, lambda=10000)", "Newton_n3_lam10000.png")
