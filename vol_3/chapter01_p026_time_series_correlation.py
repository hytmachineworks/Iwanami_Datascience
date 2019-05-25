# coding=utf-8
"""
create on : 2019/05/18
project name : Iwanami_Datascience
file name : chapter01_p026_time_series_correlation

"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

# initialize random seed
np.random.seed(seed=3)


def create_rand_walk(time_length):
    """ create random walk 1d array

    :param time_length: array length int
    :return: random walk numpy array
    """

    random_plus_minus = 2.0 * np.random.rand(time_length) - 1.0

    random_walk = np.cumsum(random_plus_minus) + np.random.rand()

    return random_walk


def main():
    """ simulate time series effect to pearson r ratio

    :return: None
    """

    times = 50

    d = pd.date_range("5 1 2019", periods=times, freq="D")

    x = create_rand_walk(times)
    y = create_rand_walk(times)

    df = pd.DataFrame(np.vstack((x, y)).T, d, columns=["X", "Y"])

    # plot 2 random walk x y

    plt.figure(figsize=(8, 8))

    sns.lineplot(data=df, palette="plasma", dashes=False)

    plt.tight_layout()

    plt.show()

    # draw scatter plot random walk x y

    plt.figure(figsize=(8, 8))

    sns.lmplot(x="X", y="Y", data=df)

    plt.tight_layout()

    plt.show()

    # simulate time series correlation

    size = 10000

    # calculate pearson r ratio at individual x y on time series
    r_indiv_list = [pearsonr(np.random.randn(times), np.random.randn(times))[0]
                    for _ in range(size)]

    min_val = sorted(r_indiv_list)[int(size * 0.025)]
    max_val = sorted(r_indiv_list, reverse=True)[int(size * 0.025)]

    # caluculate pearson r ratio at random walk x y
    r_list = [pearsonr(create_rand_walk(times),
                       create_rand_walk(times))[0]
              for _ in range(size)]

    sns.distplot(r_list, kde=False)

    plt.axvline(min_val, 0, max(r_list))
    plt.axvline(max_val, 0, max(r_list))

    plt.show()


if __name__ == "__main__":
    main()
