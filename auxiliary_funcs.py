from math import floor, pi, acos, sqrt
import random
from hyper_params import *
import torch

def value_to_bin(value, min, max, num_bins):
    if value < min:
        return 0
    elif value > max:
        return num_bins - 1
    value = value - min
    max = max - min
    step = max / (num_bins - 2)
    bin = floor(value / step) + 1
    return bin

def angle_bin_to_value(bin, min, max, num_bins, precision):
    if bin == 0:
        rge = (torch.tensor([min - 0.2]), min)
    elif bin == num_bins - 1:
        rge = (torch.tensor([max]), max + 0.2)
    else:
        bin -= 1
        step = (max - min) / (num_bins - 2)
        minim = min + step * bin
        rge = (minim, minim + step)
    num_prec_steps = floor((rge[1] - rge[0]) / precision)
    return rge[0] + precision * random.randint(0, num_prec_steps)

def speed_bin_to_value(bin, min, max, num_bins, precision):
    if bin == 0:
        rge = (torch.tensor([min]) - 0.2, min)
    elif bin == num_bins - 1:
        rge = (torch.tensor([max]), max + 0.2)
    else:
        bin -= 1
        step = (max - min) / (num_bins - 2)
        minim = min + step * bin
        rge = (minim, minim + step)
    num_prec_steps = floor((rge[1] - rge[0]) / precision)
    return rge[0] + precision * random.randint(0, num_prec_steps)


def vec_to_angle(x, y):
    if y >= 0:
        return acos(x)
    else:
        return -acos(x)


def rad_to_deg360(a):
    a %= 2 * pi
    return (a * 360) / (2 * pi)


def vec_add(u, v):
    res = []
    for i in range(len(u)):
        res.append(u[i] + v[i])
    return res


def scalar_mul(scalar, v):
    res = []
    for i in range(len(v)):
        res.append(scalar * v[i])
    return res


def ang_diff(source, target):
    a = target - source
    a = a - 2 * pi if a > pi else (a + 2 * pi if a < -pi else a)
    return a


# map distance to value in range(0,1) - greater distance -> lesser intensity (from Moritz Maxeiner)
def intensity_linear(distance):
    if distance < 0 or far_plane < distance:
        return 0
    return 1 - float(distance) / far_plane


def intensity_angular(ang_diff):
    return 1 - float(ang_diff) / pi


def addCorner(p1, p2):
    x = p2[0] if 0 < p1[0] < 100 else p1[0]
    y = p2[1] if 0 < p1[1] < 100 else p1[1]
    return x, y


def dist(p, q):
    return sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def dot_product(vec1, vec2):
    akk = 0
    for i in range(len(vec1)):
        akk += vec1[i] * vec2[i]
    return akk
