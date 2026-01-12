from __future__ import annotations

import math
from typing import List
import numpy as np

from floretion import Floretion


def get_base_vec_centroid_dist(base_vector_oct: str) -> float:
    """
    Distanza del centroide (algoritmo triangoli) dal punto (0,0) con passo iniziale 1.
    """
    x = 0.0
    y = 0.0
    distance = 1.0
    sign_distance = -1.0

    for digit in base_vector_oct:
        if digit == "7":
            sign_distance *= -1.0
        else:
            if digit == "4":
                angle = 210
            elif digit == "2":
                angle = 90
            elif digit == "1":
                angle = 330
            else:
                raise ValueError(f"Invalid digit {digit} in base vector.")
            x += math.cos(math.radians(angle)) * distance * sign_distance
            y += math.sin(math.radians(angle)) * distance * sign_distance
        distance /= 2.0

    return math.hypot(x, y)


def get_basevec_coords(base_vector_oct: str) -> List[float]:
    """
    Coordinate (x,y) del centroide usando passo iniziale 1.
    """
    x = 0.0
    y = 0.0
    distance = 1.0
    sign_distance = -1.0

    for digit in base_vector_oct:
        if digit == "7":
            sign_distance *= -1.0
        else:
            if digit == "4":
                angle = 210
            elif digit == "2":
                angle = 90
            elif digit == "1":
                angle = 330
            else:
                raise ValueError(f"Invalid digit {digit} in base vector.")
            x += math.cos(math.radians(angle)) * distance * sign_distance
            y += math.sin(math.radians(angle)) * distance * sign_distance
        distance /= 2.0

    return [x, y]


def clip_coeffs(floretion: Floretion, clip_threshold: float) -> Floretion:
    """
    Azzeramento coefficienti i cui base vectors hanno distanza centroide > clip_threshold.
    """
    coeff_array = []
    for base_vector, coeff in zip(floretion.grid_flo_loaded_data["oct"], floretion.coeff_vec_all):
        distance_to_unit = get_base_vec_centroid_dist(base_vector)
        if distance_to_unit > float(clip_threshold):
            coeff_array.append(0.0)
        else:
            coeff_array.append(float(coeff))
    coeff_array_final = np.array(coeff_array, dtype=float)
    return Floretion(coeff_array_final, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)
