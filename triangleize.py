# MIT License
# Copyright (c) [2025 [Creighton Dement]
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Tuple, Optional
import math

import cv2
import numpy as np

from floretion import Floretion






from lib.triangleize_utils.coloring import (
    ColorMode,
    choose_max_val_for_colors,
    map_color,
    NegPolicy
)



class Triangleize:
    """
    Rappresentazione Sierpinski-based con triangoli o punti.
    """

    def __init__(self, floretion_object: Floretion, image: np.ndarray, plot_type: str = "triangle", distance_scale_fac: float = 4.0):
        self.floretion = floretion_object
        self.base_vectors = self.floretion.grid_flo_loaded_data["oct"]
        self.coeffs = self.floretion.coeff_vec_all

        self.img = image
        self.plot_type = plot_type
        self.height, self.width = self.img.shape[0], self.img.shape[1]
        self.distance_scale_fac = float(distance_scale_fac)

    def draw_dot(self, x: float, y: float, brightness: float, color_bgr: Tuple[int, int, int]) -> None:
        scaled_color = tuple(int(max(0, min(255, c * brightness))) for c in color_bgr)
        radius = max(1, int(self.height * 0.003))
        cv2.circle(self.img, (int(x), int(y)), radius, scaled_color, -1)

    def draw_triangle(self, x: float, y: float, height: float, orientation: str, brightness: float, color_bgr: Tuple[int, int, int]) -> None:
        half_base = math.sin(math.pi / 3.0) * height
        if orientation == "up":
            vertices = np.array([[x, y - height], [x - half_base, y + height / 2.0], [x + half_base, y + height / 2.0]], dtype=np.int32)
        else:
            vertices = np.array([[x, y + height], [x - half_base, y - height / 2.0], [x + half_base, y - height / 2.0]], dtype=np.int32)
        scaled_color = tuple(int(max(0, min(255, c * brightness))) for c in color_bgr)
        cv2.fillConvexPoly(self.img, vertices.reshape((-1, 1, 2)), scaled_color)

    @staticmethod
    def calculate_orientation(base_vector_oct: str) -> str:
        count = sum(1 for d in base_vector_oct if d in "124")
        order = len(base_vector_oct)
        if count % 2 == 0:
            return "up" if order % 2 == 0 else "down"
        return "down" if order % 2 == 0 else "up"

    def place_base_vecs(self, base_vector_oct: str) -> Tuple[float, float, float, float, float, np.ndarray]:
        x = self.width // 2
        y = self.height // 2
        y += int(self.height * 0.1)

        distance = self.height / self.distance_scale_fac
        x_theo = 0.0
        y_theo = 0.0
        sign_distance = -1.0

        green_value = red_value = blue_value = 0

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
                    raise ValueError(f"Cifra ottale non valida: {digit}")

                dx = math.cos(math.radians(angle)) * distance * sign_distance
                dy = math.sin(math.radians(angle)) * distance * sign_distance
                x += dx
                y += dy
                x_theo += dx
                y_theo += dy

            green_distance = math.hypot(x - 0.0, y - 0.0)
            red_distance = math.hypot(x - float(self.width), y - 0.0)
            blue_distance = math.hypot(x - float(self.width) / 2.0, y - float(self.height))

            max_distance = math.hypot(float(self.height), float(self.width))
            green_value = int((green_distance / max_distance) * 255.0)
            red_value = int((red_distance / max_distance) * 255.0)
            blue_value = int((blue_distance / max_distance) * 255.0)

            distance /= 2.0

        color_array = np.array([red_value, green_value, blue_value], dtype=np.int32)
        return x, y, x_theo, y_theo, 1.3 * distance, color_array

    def plot_floretion(
            self,
            title: Optional[str] = None,
            highlight_base_vec: Optional[int] = None,
            *,
            color_mode: ColorMode = "abs-hsv",
            max_val: Optional[float] = None,  # se vuoi assumere “2” metti default=2.0
            auto_clip_pct: float = 99.0,
            gamma: float = 0.6,
            sat_dist_weight: float = 0.5,
            neg_policy: NegPolicy = "hue-180",
            band_count: int = 8,
    ) -> None:
        """
        Colori robusti per coefficienti grandi, delega a lib.triangleize_utils.coloring.
        """
        if title:
            cv2.putText(self.img, title, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # Valore massimo “robusto”
        max_val_eff = choose_max_val_for_colors(self.floretion.coeff_vec_all, max_val, auto_clip_pct)

        num_basevecs = float(len(self.floretion.coeff_vec_all))
        index = 0

        for base_vector, coeff in zip(self.base_vectors, self.floretion.coeff_vec_all):
            x, y, x_theo, y_theo, final_distance, _ = self.place_base_vecs(str(base_vector))

            basevec_at_pct = index / (num_basevecs - 1.0) if num_basevecs > 1 else 0.0
            dist_norm = math.hypot(x_theo, y_theo) / max(1.0, float(self.height))

            color, brightness01 = map_color(
                coeff=float(coeff),
                basevec_at_pct=basevec_at_pct,
                dist_norm=dist_norm,
                max_val=max_val_eff,
                mode=color_mode,
                gamma=gamma,
                sat_dist_weight=sat_dist_weight,
                neg_policy=neg_policy,
                band_count=band_count,
            )

            if highlight_base_vec is not None and int(base_vector) == int(highlight_base_vec):
                color = (255, 255, 255)
                brightness01 = 1.0

            orientation = self.calculate_orientation(str(base_vector))

            if self.plot_type in ("dot", "dots"):
                self.draw_dot(x, y, brightness01, color)
            elif self.plot_type in ("triangle", "triangles"):
                self.draw_triangle(x, y, final_distance, orientation, brightness01, color)
            else:
                raise ValueError(f"Unknown plottype {self.plot_type}")

            index += 1


# Demo veloce eseguibile direttamente

if __name__ == "__main__":

    W, H = 2024, 2024
    from lib.triangleize_utils.ripples import render_radial_ripples_sequence
    from lib.triangleize_utils.centroid_distance import flo_from_centroid_distance

    order = 7
    #seed = flo_from_centroid_distance(order=order, pct=100, relation="le", coeff="dist")
    seed = Floretion.get_typical_floretions("sierp_flo", order)
    out_dir, frames = render_radial_ripples_sequence(
        width=W, height=H,
        order=order, total_frames=1200,
        seed=seed,
        wavelength_pct=8.0,
        speed_pct_per_frame=0.1,
        amplitude=0.8, bias=0.2, phase0=0.0, damping_alpha=0.5,
        mix="mul",
        normalize_to=2.0,
        color_mode="log-hsv",  # oppure "legacy" per il look classico
        output_dirname="yeah_baby_now"
    )
    print("frames in:", out_dir)


