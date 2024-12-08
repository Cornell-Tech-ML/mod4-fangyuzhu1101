import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of data points where each point is a tuple containing
    two random floating-point numbers as coordinates.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        A list of tuples, each containing two float values representing coordinates.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset where the class of each point depends on whether the
    x-coordinate (x_1) is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the simple dataset with number of points to
        generate (N), coordinates (X), and binary class labels (y) based on a simple split.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a diagonal dataset where the class of each point depends on
    whether the sum of x and y coordinates (x_1 + x_2) is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the diagonal dataset with number of points to
        generate (N), coordinates (X), and binary class labels (y) based on a diagonal split.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a split dataset where the class of each point depends on whether
    the x-coordinate (x_1) being either less than 0.2 or greater than 0.8.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the split dataset with number of points to generate (N),
        coordinates (X), and binary class labels (y) where each point is classified based on its x-coordinate.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a XOR dataset where the class of each point depends on whether the x (x_1) and y (x_2)
    coordinates satisfying the XOR condition ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the XOR dataset with number of points to
        generate (N), coordinates (X), and binary class labels (y) based on a XOR split.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a circle dataset where the class of each point depends on whether it
    lies outside a circle centered at (0.5, 0.5) with the radius approximately 0.3162 (sqrt(0.1)).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the circle dataset with number of points to generate (N),
        coordinates (X), and binary class labels (y) based on a circle split, specific distance from the center.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral dataset where two interlocking spiral patterns are formed,
    each belonging to a different class.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        An instance of the Graph class containing the spiral dataset with number of points to
        generate (N), coordinates (X), and binary class labels (y) where each point is classified
        based on a spiral split, forming two spirals, classified into two categories.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
