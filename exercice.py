#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import math


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, num=64)


def coordinate_conversion(cartesian_coordinates: list[tuple]) -> list[tuple]:
    return [(np.hypot(coord[0], coord[1]), sum(np.arctan(coord)))
            for coord in cartesian_coordinates]


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def plot_function() -> None:
    x = np.linspace(-1, 1, num=250)
    y = x**2 * np.sin(1 / x**2) + x
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def estimate_pi(sampleSize: int):
    coordinates = np.random.random((sampleSize, 2))
    inCircle = coordinates[:,0]**2 + coordinates[:,1]**2 < 1
    plt.plot(coordinates[inCircle][:,0], coordinates[inCircle][:,1], "bo")
    plt.plot(coordinates[inCircle == False][:,0], coordinates[inCircle == False][:,1], "ro")
    plt.title("Calcul de π par la méthode de Monte Carlo.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return 4 * coordinates[inCircle].size / coordinates.size


def evaluate_integral(function):
    return integrate.quad(function, -np.inf, np.inf)[0]


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
    print(coordinate_conversion([(1, 0),
                                 (0, 1),
                                 (1, 1)]))
    print(find_closest_index(np.array([0,1,2,3,4,5]), 3))
    plot_function()
    print(estimate_pi(5000))
    print(evaluate_integral(lambda x: np.exp(-x**2)))
