# coding=utf-8
"""
create on : 2019/05/15
project name : Iwanami_Datascience
file name : chapter01_p018_stratified_analysis

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
    """ comparison whole vs stratified analysis

    :return: None
    """

    # average
    mu = [0, 0]

    # r ratio
    r2 = 0.8

    # sample size
    size = 10000

    # stratified count
    stratify = 4

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

    # divide z by equal quantity
    df["z_div"] = pd.qcut(df["z"], stratify)

    # decorate legend
    z_div_list = list(set(list(df["z_div"])))
    z_div_dict = {z_div_val: "item no." + str(z_div_list.index(z_div_val))
                  for z_div_val in z_div_list}

    df["z_div"] = df["z_div"].apply(lambda z_x: z_div_dict[z_x])

    plt.figure(figsize=(8, 8))

    sns.relplot(x="x", y="y", hue="z_div", hue_order=z_div_dict.values(),
                alpha=.5, palette="plasma", data=df)

    # get coef by OLS
    results = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()

    org_dict = {"x": results.params["x"],
                "const": results.params["const"]}

    plot_x = np.linspace(min(x), max(x), 2)

    plt.plot(plot_x, plot_x * x_coef_true, "k-")

    whole_y = org_dict["x"] * plot_x + org_dict["const"]

    x_list = []
    x_weight_list = []

    const_list = []
    const_weight_list = []

    for z_dict_key in list(z_div_dict.values()):
        res = sm.OLS(df[df["z_div"] == z_dict_key]["y"],
                     sm.add_constant(df[df["z_div"] == z_dict_key]["x"])).fit()

        n_buff = df[df["z_div"] == z_dict_key].shape[0]

        # calculate variation
        x_std_var_buff = (res.bse["x"] ** 2) * n_buff
        const_std_var_buff = (res.bse["const"] ** 2) * n_buff

        # calculate weight
        x_weight_buff = 1 / x_std_var_buff
        const_weight_buff = 1 / const_std_var_buff

        x_list.append(res.params["x"] * x_weight_buff)
        x_weight_list.append(x_weight_buff)

        const_list.append(res.params["const"] * const_weight_buff)
        const_weight_list.append(const_weight_buff)

    # calculate weighted average
    x_chef_stratified = sum(x_list)/sum(x_weight_list)
    const_stratified = sum(const_list)/sum(const_weight_list)

    stratified_y = x_chef_stratified * plot_x + const_stratified

    print("-"*10)

    print("calculate all coef and cont values")
    print("x coef : {x}, const : {const}".format(x=str(org_dict["x"]),
                                                 const=str(org_dict["const"])))

    plt.plot(plot_x, whole_y, "r--")

    print("-"*10)

    print("calculate all coef and cont values")
    print("x coef : {x}, const : {const}".format(x=str(x_chef_stratified),
                                                 const=str(const_stratified)))

    plt.plot(plot_x, stratified_y, "g--")

    print("-"*10)

    print("true x coef : {}".format(str(x_coef_true)))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
