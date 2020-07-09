from os import listdir
from os.path import join, isfile

import h5py
import numpy
from math import *
from matplotlib import pyplot
from matplotlib import patches
from shapely.geometry import Polygon as ShapelyPolygon
import shapely
from descartes.patch import PolygonPatch
from figures import BLUE, RED, BLACK, YELLOW, SIZE, set_limits, plot_coords, color_isvalid
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
from hyper_params import *
import os
from guppy_model import LSTM_fixed, LSTM_multi_modal
import torch


# Whole code is inspired by Moritz Maxeiners master thesis and his code for the thesis


# some simple helper functions

# convert a unit-vector giving the orientation to its angle to the x-axis (in radians)
def vec_to_angle(x, y):
    if y >= 0:
        return acos(x)
    else:

        return -acos(x)


def rad_to_deg360(a):
    # if a < 0:
    #    a = 2 * pi - a

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


max_dist = 70


def ang_diff(source, target):
    a = target - source
    a = a - 2 * pi if a > pi else (a + 2 * pi if a < -pi else a)
    return a


# map distance to value in range(0,1) - greater distance -> lesser intensity (from Moritz Maxeiner)
def intensity_linear(distance):
    if distance < 0 or max_dist < distance:
        return 0
    return 1 - float(distance) / max_dist


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


def get_locomotion_vec(data_prev, data_cur):
    if data_prev is None:
        return 0, 0
    else:
        # ang_turn = vec_to_angle(data_prev[2], data_prev[3]) - vec_to_angle(data_cur[2], data_cur[3])
        ang_turn = ang_diff(vec_to_angle(data_prev[2], data_prev[3]), vec_to_angle(data_cur[2], data_cur[3]))
        diff_vec = ((data_cur[0] - data_prev[0]), (data_cur[1] - data_prev[1]))
        linear_speed = dot_product(diff_vec, (data_cur[2], data_cur[3]))
        return ang_turn, linear_speed


def dist_travelled(data_prev, data_cur):
    if data_prev is None:
        return 0, 0
    else:
        return sqrt((data_cur[0] - data_prev[0]) ** 2 + (data_cur[1] - data_prev[1]) ** 2)


def is_in_tank(x, y):
    return 0 <= x <= 100 and 0 <= y <= 100


tank_walls = [((1, 0), (0, 0)), ((1, 0), (0, 100)), ((0, 1), (0, 0)), ((0, 1), (100, 0))]


def ray_intersection(x, a):
    # we have the following equations:
    # lambda * cos(a) + x1 = mu * w1 + b1
    # lambda * sin(a) + x2 = mu * w2 + b2
    # x = (x.x, x.y)
    x = list(x)

    # make sure positions are within tank walls
    x[0] = 0 if x[0] < 0 else x[0]
    x[1] = 0 if x[1] < 0 else x[1]
    x[0] = 100 if x[0] > 100 else x[0]
    x[1] = 100 if x[1] > 100 else x[1]

    a = a % (2 * pi)
    c = cos(a)
    s = sin(a)
    # if we have a case where cos or sin are close to 0, we can immediately return some value
    if abs(c) < 0.0001:
        # print('edge case: angle almost 90 or 270')
        if a > 0:
            return x[0], 100
        else:
            return x[0], 0
    elif abs(s) < 0.0001:
        if abs(a) < 0.9:
            return 100, x[1]
        else:
            return 0, x[1]

    # else we iterate through the lines which represent the tank walls
    # and search the intersection point with the line given by position and orientation-angle of the agent
    # there has to be exactly one intersection point within the tank walls in the direction of the ray
    for w, b in tank_walls:
        lambd = 0
        if w[0] == 0:
            lambd = (b[0] - x[0]) / c
        elif w[1] == 0:
            lambd = (b[1] - x[1]) / s

        # we want the intersection in the direction of the ray, so lambda has to be bigger than 0
        if lambd >= 0:
            inters = round(lambd * c + x[0], 6), round(lambd * s + x[1], 6)
            if is_in_tank(inters[0], inters[1]):
                return inters
    print("ERROR: no intersection point found!")
    print("Position: ", x[0], x[1])
    print("Orientation: ", a)
    print("cos(a): ", c, " sin(a): ) ", s)
    return -1, -1


def get_bin(value, min, max, num_bins):
    if value < min:
        return 0
    elif value > max:
        return num_bins - 1
    value = value - min
    max = max - min
    step = max / (num_bins - 2)
    bin = floor(value / step) + 1
    return bin


def normalize_ori(ori):
    if ori[0] > 1: ori[0] = 1
    if ori[0] < -1: ori[0] = -1
    if ori[1] > 1: ori[1] = 1
    if ori[1] < -1: ori[1] = -1
    return ori


def normalize_pos(pos):
    if pos[0] < 0: pos[0] = 0
    if pos[1] < 0: pos[1] = 0
    if pos[0] > 100: pos[0] = 100
    if pos[1] > 100: pos[1] = 100
    return pos


# thats the main class, its run method will do all the work
class Guppy_Calculator():
    def __init__(self, filepath, agent, num_guppy_bins, num_wall_rays, livedata, simulation=False):
        # set up data

        self.filepath = filepath
        with h5py.File(filepath, "r") as hdf_file:
            self.ls = list(hdf_file.keys())
            self.num_guppys = len(self.ls)
            if not livedata:
                self.data = [numpy.array(hdf_file.get("{}".format(i))) for i in range(self.num_guppys)]
            else:
                self.data = [numpy.array(hdf_file.get("{}".format(i + 1))) for i in range(self.num_guppys)]
            self.agent = agent
            self.agent_data = self.data[agent]

        self.length = len(self.agent_data)

        self.num_bins = num_guppy_bins
        self.num_rays = num_wall_rays
        self.bin_angles = [pi * (i / self.num_bins) - pi / 2 for i in range(0, self.num_bins + 1)]
        self.wall_angles = [pi * (i / self.num_rays) - pi / 2 for i in range(0, self.num_rays)]
        self.simulation = simulation
        if simulation:
            self.fig, self.ax = pyplot.subplots()
            self.ax.set_title('tank')
            set_limits(self.ax, 0, 100, 0, 100)

    def get_data(self, agent):
        self.agent_data = self.data[agent]
        self.length = len(self.agent_data)
        sensory = []
        loc = []
        for i in range(1, len(self.agent_data)):
            loc.append(self.get_loc_vec(i))
            sensory.append(self.craft_vector(i, agent))
        return numpy.array(loc), numpy.array(sensory)

    def network_simulation(self):
        # load model
        torch.set_default_dtype(torch.float64)
        path = network_path
        state_dict = torch.load(path)
        hidden_layer_size = state_dict["lstm.weight_hh_l0"][1]
        model = LSTM_fixed()
        model.load_state_dict(state_dict)
        model.eval()

        # init hidden
        hidden_state = [model.init_hidden(1, num_layers) for agent in range(self.num_guppys)]
        for i in range(1, len(self.agent_data) - 1):
            for agent in range(self.num_guppys):
                with torch.no_grad():
                    # get input data for this frame
                    sensory = self.craft_vector(i, agent)
                    data = torch.from_numpy(numpy.concatenate((self.loc_vec, sensory)))
                    data = data.view(1, 1, -1)

                    # predict the new ang_turn, lin_speed
                    out, hidden_state[agent] = model.predict(data, hidden_state[agent])
                    ang_turn = out[0][0][0].item()
                    lin_speed = out[0][0][1].item()

                    # rotate agent position by angle calculated by network
                    cos_a = cos(ang_turn)
                    sin_a = sin(ang_turn)
                    agent_pos = self.data[agent][i][0], self.data[agent][i][1]
                    agent_ori = self.data[agent][i][2], self.data[agent][i][3]
                    new_ori = [cos_a * agent_ori[0] - sin_a * agent_ori[1], \
                               sin_a * agent_ori[0] + cos_a * agent_ori[1]]
                    # normally the rotation of a normalized vector by a normalized vector should again be a
                    # normalized vector, but it seems there are some numerical errors, so normalize the orientation
                    # again
                    normalize_ori(new_ori)

                    # multiply new orientation by linear speed and add to old position
                    translation_vec = scalar_mul(lin_speed, new_ori)
                    new_pos = vec_add(agent_pos, translation_vec)
                    # network does not learn the tank walls properly sometimes, let fish bump against the wall
                    normalize_pos(new_pos)

                    # update the position for the next timestep
                    self.data[agent][i + 1][0], self.data[agent][i + 1][1] = new_pos
                    self.data[agent][i + 1][2], self.data[agent][i + 1][3] = new_ori

            self.plot_guppy_bins(bins=False)

    # for estimating the bin boundaries, we want know no the maximum and minimum angular turn and linear speed
    # that occurred for this file
    def get_min_max_angle_speed(self):
        angle, speed = self.get_loc_vec(1)
        min_angle = max_angle = angle
        min_speed = max_speed = speed
        for i in range(2, len(self.agent_data)):
            angle, speed = self.get_loc_vec(i)
            min_angle = angle if angle < min_angle else min_angle
            max_angle = angle if angle > max_angle else max_angle
            min_speed = speed if speed < min_speed else min_speed
            max_speed = speed if speed > max_speed else max_speed
        return min_angle, max_angle, min_speed, max_speed

    def craft_vector(self, i, agent):
        """calculate the vector v = (locomotion, agent_view, wall_view) from the raw data
        for the given agent at given frame"""
        # get position of guppy 0 and convert it to sympy point
        self.agent_data = self.data[agent]
        self.obs_pos = (self.agent_data[i][0], self.agent_data[i][1])
        # get orientation vector
        self.obs_ori = (self.agent_data[i][2], self.agent_data[i][3])
        # calculate angle of guppy_orientation with respect to x_axis
        self.obs_angle = vec_to_angle(self.obs_ori[0], self.obs_ori[1])
        # calculate positions of other guppys
        self.others = [(self.data[j][i][0], self.data[j][i][1])
                       for j in range(self.num_guppys) if j != agent]
        self.others_angle = [vec_to_angle(self.data[j][i][2], self.data[j][i][3])
                             for j in range(self.num_guppys) if j != agent]
        # calculate intensity vector wall_view for distance to walls
        self.wall_distances()
        # calculate intensity vector agent_view for distance to nearby guppies
        self.guppy_distances()
        # loc_vec[0] = angular turn; loc_vec[1] = linear speed
        if self.simulation:
            self.loc_vec = get_locomotion_vec(self.agent_data[i - 1], self.agent_data[i])

        # we return the vector already concatenated in a numpy_vector
        if include_others_angles:
            return numpy.concatenate((numpy.array(self.agent_view),
                                      numpy.array(self.wall_view),
                                      numpy.array(self.agent_view_angle)))
        else:
            return numpy.concatenate((numpy.array(self.agent_view),
                                      numpy.array(self.wall_view)))

    # get the locomotion vector for frame i
    def get_loc_vec(self, i):
        return numpy.array(get_locomotion_vec(self.agent_data[i - 1], self.agent_data[i]))

    # visualize trajectories and bins for a whole file
    def run_sim(self, step):
        for frame in range(1, len(self.agent_data), step):
            self.craft_vector(frame, self.agent)
            self.dist_difference = dist_travelled(self.agent_data[frame - 1], self.agent_data[frame])
            self.plot_guppy_bins()
            # self.plot_wall_rays() #use either of the two plot_functions, not both at once

    def guppy_distances(self):
        # get the intersection points of the rays with the tank walls
        intersections = [ray_intersection(self.obs_pos, angle + self.obs_angle) for angle in self.bin_angles]
        # construct the bins as polygons with intersection points and observer position as vertices
        # would surely be more efficient to just use a function which checks if a point lies within a polygon defined
        # by the three / four points given.
        self.bins = []
        for i in range(len(intersections) - 1):
            if intersections[i][0] == intersections[i + 1][0] or intersections[i][1] == intersections[i + 1][1]:
                self.bins.append(ShapelyPolygon([self.obs_pos, intersections[i], intersections[i + 1]]))
            else:  # if the intersection points overlap a corner of the tank, we have to add that corner to the polygon
                corner = addCorner(intersections[i], intersections[i + 1])
                self.bins.append(ShapelyPolygon([self.obs_pos, intersections[i], corner, intersections[i + 1]]))

        # loop through the bins and find the closest guppy for each bin
        self.agent_view = [1000.0 for i in range(len(self.bins))]
        self.agent_view_angle = [0 for i in range(len(self.bins))]
        length = len(self.others)
        others_c = self.others[:]
        others_ang = self.others_angle[:]

        # Variant 1: Start with the bins and delete guppys which already found their bin
        # This seems to be the most efficient
        for i in range(len(self.bins)):
            j = 0
            while j < length:
                if self.bins[i].contains(shapely.geometry.Point(others_c[j][0], others_c[j][1])):
                    distance = dist(self.obs_pos, others_c[j])
                    if distance < self.agent_view[i]:
                        self.agent_view[i] = distance
                        ang_dif = abs(self.obs_angle - others_ang[j])
                        if ang_dif > pi:
                            ang_dif = 2 * pi - ang_dif
                        self.agent_view_angle[i] = intensity_angular(ang_dif)
                    del (others_c[j])
                    del (others_ang[j])
                    length -= 1
                else:
                    j += 1

            self.agent_view[i] = intensity_linear(self.agent_view[i])
        # agent_view vector is ready now

    def plot_guppy_bins(self, bins=True):
        self.ax.cla()  # clear axes
        # plot tank
        ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
        polygon = shapely.geometry.Polygon(ext)
        plot_coords(self.ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.6, zorder=1)
        self.ax.add_patch(patch)
        # plot bins
        if bins:
            for i, p in enumerate(self.bins):
                patch = PolygonPatch(p, facecolor=BLACK, edgecolor=YELLOW,
                                     alpha=(1 - self.agent_view[i]), zorder=1)
                self.ax.add_patch(patch)
        # plot fishes
        ellipse = patches.Ellipse((self.obs_pos[0], self.obs_pos[1]), 2, 7, rad_to_deg360(self.obs_angle - pi / 2))
        self.ax.add_patch(ellipse)
        for i in range(len(self.others)):
            position = (self.others[i][0], self.others[i][1])
            ellipse = patches.Ellipse(position, 2, 7, rad_to_deg360(self.others_angle[i] - pi / 2))
            self.ax.add_patch(ellipse)
        # plot legend
        self.ax.text(110, 100, 'angle: {:.2f}°'.format(degrees(self.obs_angle)))
        self.ax.text(110, 90, 'o_vector: ({:.2f},{:.2f})'.format(self.obs_ori[0], self.obs_ori[1]))
        self.ax.text(110, 80, 'angular turn: {:.10f}'.format(self.loc_vec[0]))
        self.ax.text(110, 70, 'linear speed: {:.10f}'.format(self.loc_vec[1]))
        # self.ax.text(110, 60, 'dist travelled: {:.10f}'.format(self.dist_difference))
        pyplot.show(block=False)
        pyplot.pause(0.00000000000001)

    def wall_distances(self):
        # self.intersections = [ray_intersection(self.obs_pos, angle + self.obs_angle) for angle in self.wall_angles]
        self.wall_view = [intensity_linear(
            dist(self.obs_pos, ray_intersection(self.obs_pos, angle + self.obs_angle)))
            for angle in self.wall_angles]

    def plot_wall_rays(self):
        self.ax.cla()
        # plot tank
        ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
        polygon = shapely.geometry.Polygon(ext)
        plot_coords(self.ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.6, zorder=1)
        self.ax.add_patch(patch)
        # plot rays
        self.ax.plot(self.obs_pos[0], self.obs_pos[1], "ro")
        # uncomment line for intersections in wall_distances
        for (x, y) in self.intersections:
            self.ax.plot([self.obs_pos[0], x], [self.obs_pos[1], y])

        pyplot.show(block=False)
        pyplot.pause(0.00000000001)


class Guppy_Dataset(Dataset):
    def __init__(self, filepaths, agent, num_bins, num_rays, livedata, output_model):
        self.livedata = livedata
        self.num_rays = num_rays
        self.num_view_bins = num_bins
        self.agent = agent
        # self.length = len(filepaths)
        self.length = 0
        self.filepaths = filepaths
        self.data = []
        self.datapaths = []
        self.labelpaths = []

        # preprocess data for each file given by filepaths
        for i in range(len(filepaths)):
            datapath = self.filepaths[i]
            labelpath = self.filepaths[i]
            datapath += "data.{}".format(output_model)
            labelpath += "label.{}".format(output_model)
            gc = Guppy_Calculator(self.filepaths[i], self.agent, self.num_view_bins, self.num_rays, self.livedata)
            self.length += gc.num_guppys
            # get processed data from the perspective of guppy of a file
            for agent in range(gc.num_guppys):
                final_data_path = datapath + "ag" + str(agent) + ".npy"
                final_label_path = labelpath + "ag" + str(agent) + ".npy"
                if not os.path.isfile(final_data_path):
                    x_loc, x_sensory = gc.get_data(agent)
                    y_loc = numpy.roll(x_loc, -1, 0)
                    if output_model == ("multi_modal"):
                        for i in range(y_loc.shape[0]):
                            y_loc[i, 0] = get_bin(y_loc[i, 0], angle_min, angle_max, num_angle_bins)
                            y_loc[i, 1] = get_bin(y_loc[i, 1], speed_min, speed_max, num_speed_bins)
                    numpy.save(final_data_path, numpy.concatenate((x_loc[:-1, :], x_sensory[:-1, :]), 1))
                    numpy.save(final_label_path, y_loc[:-1, :])
                self.datapaths.append(final_data_path)
                self.labelpaths.append(final_label_path)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        data = numpy.load(self.datapaths[i])
        labels = numpy.load(self.labelpaths[i])
        return data, labels


if __name__ == "__main__":
    trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
    files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5")]
    files.sort()
    num_files = len(files)
    files = files[:num_files]
    print(files)
    filepath = "guppy_data/couzin_torus/train/8_0002.hdf5"
    filepathlive = "guppy_data/live_female_female/train/CameraCapture2019-06-28T15_40_01_9052-sub_3.hdf5"
    gc = Guppy_Calculator(filepathlive, agent=0,
                          num_guppy_bins=num_guppy_bins,
                          num_wall_rays=num_wall_rays,
                          livedata=live_data, simulation=True)
    # gc.network_simulation()
    gc.run_sim(step=1)
