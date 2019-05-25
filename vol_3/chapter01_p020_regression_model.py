# coding=utf-8
"""
create on : 2019/05/17
project name : Iwanami_Datascience
file name : chapter01_p020_regression_model

"""
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import multivariate_normal
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# initialize random seed
np.random.seed(seed=3)


def main():
    """ estimate coef x by multiple regression model

    :return: None
    """

    # average
    mu = [0, 0]

    # r ratio
    r2 = 0.8

    # sample size
    size = 10000

    x_coef_true = 1.5

    sigma = np.array([[1, r2],
                      [r2, 1]])

    # create 2d normal random matrix to x, z
    values = multivariate_normal(mu, sigma, size)

    print("x, z correlation coef : ", np.corrcoef(values[:, 0], values[:, 1]))

    x = values[:, 0]
    z = values[:, 1]

    # create random normal distribution factor
    e = np.random.normal(0.0, 1.0, size)

    # calculate y
    y = x_coef_true * x + 1.1 * z + e

    df = pd.DataFrame([x, y, z]).T
    df.columns = ["x", "y", "z"]

    plt.figure(figsize=(8, 8))

    sns.relplot(x="x", y="y", alpha=.5, palette="plasma", data=df)

    plot_x = np.linspace(min(x), max(x), 2)

    plt.plot(plot_x, plot_x * x_coef_true, "k-")

    # get coef x by OLS
    results_xy = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()

    org_dict = {"x": results_xy.params["x"],
                "const": results_xy.params["const"]}

    print(results_xy.summary())

    plot_y = org_dict["x"] * plot_x + org_dict["const"]

    # get coef by OLS
    results_xyz = sm.OLS(df["y"], sm.add_constant(df[["x", "z"]])).fit()

    xyz_dict = {"x": results_xyz.params["x"],
                "const": results_xyz.params["const"]}

    print(results_xyz.summary())

    plot_x = np.linspace(min(x), max(x), 2)
    plot_yz = xyz_dict["x"] * plot_x + xyz_dict["const"]

    print("-"*10)

    print("calculate all coef and cont values")
    print("x coef : {x}, const : {const}".format(x=str(org_dict["x"]),
                                                 const=str(org_dict["const"])))

    plt.plot(plot_x, plot_y, "r--")

    print("-"*10)

    print("calculate all coef and cont values")
    print("x coef : {x}, const : {const}".format(x=str(xyz_dict["x"]),
                                                 const=str(xyz_dict["const"])))

    plt.plot(plot_x, plot_yz, "g--")

    print("-"*10)

    print("true x coef : {}".format(str(x_coef_true)))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
