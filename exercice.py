#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    angles = np.sum(np.arctan(cartesian_coordinates), axis=1)
    radius = np.hypot(cartesian_coordinates[:, 0], cartesian_coordinates[:, 1])
    print(angles)
    print(radius)
    return np.concatenate([angles, radius], axis=1)


def find_closest_index(values: np.ndarray, number: float) -> int:
    return 0


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
    print(coordinate_conversion(np.array([[1, 0],
                                          [0, 1],
                                          [1, 1]])))
