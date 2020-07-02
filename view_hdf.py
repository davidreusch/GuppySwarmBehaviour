import h5py
import numpy
from math import *
from matplotlib import pyplot
from shapely.geometry import Polygon as ShapelyPolygon
import shapely
from descartes.patch import PolygonPatch
from figures import BLUE, RED, BLACK, YELLOW, SIZE, set_limits, plot_coords, color_isvalid
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
from hyper_params import *
import os


# Whole code is inspired by Moritz Maxeiners master thesis and his code for the thesis


# some simple helper functions

# convert a unit-vector giving the orientation to its angle to the x-axis (in radians)
def vec_to_angle(x, y):
    if y >= 0:
        return acos(x)
    else:

        return -acos(x)


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
    c = cos(a)
    s = sin(a)
    # if we have a case where cos or sin are close to 0, we can immediately return some value
    if abs(c) < 0.0001:
        # print('edge case: angle almost 90 or 270')
        if a > 0:
            return x[0], 100
        else:
            return x[0], 0
    elif abs(a) < 0.0001:
        # print('edge case: angle very small')
        return 100, x[1]
    elif abs(abs(a) - pi) < 0.0001:
        # print('edge case: angle almost 180')
        return 0, x[1]

    # else we iterate through the lines which represent the tank walls
    # and search the intersection point with the line given by position and orientation-angle of the agent
    for w, b in tank_walls:
        lambd = 0
        if w[0] == 0:
            lambd = (b[0] - x[0]) / c
        elif w[1] == 0:
            lambd = (b[1] - x[1]) / s

        if lambd >= 0:
            inters = round(lambd * c + x[0], 6), round(lambd * s + x[1], 6)
            if is_in_tank(inters[0], inters[1]):
                return inters
    print("ERROR: no intersection point found!")
    return -1, -1


def get_bin(value, min, max, num_bins):
    step = (max - min) / num_bins
    # res = np.zeros(num_bins)
    for i in range(num_bins):
        if min + i * step <= value < min + (i + 1) * step:
            # the loss function just wants the index of the correct class
            return i
    if value < min:
        return 0
    elif value > max:
        return num_bins - 1
    else:
        return print("ERROR no bin found")


# def one_hot_wrap(arr):
#    # take one hot of the first to values of the array and return these class numbers as an array
#    angle_label = get_bin(arr[0], -0.04, 0.04, angle_bins)
#    speed_label = get_bin(arr[1], 0.0, 0.4, speed_bins)
#    return numpy.array([angle_label, speed_label])

# thats the main class, its run method will do all the work
class Guppy_Calculator():
    def __init__(self, filepath, agent, num_guppy_bins, num_wall_rays, livedata, simulation=False):
        # set up data

        self.filepath = filepath
        with h5py.File(filepath, "r") as hdf_file:
            self.ls = list(hdf_file.keys())
            if not livedata:
                self.data = [numpy.array(hdf_file.get("{}".format(i))) for i in range(len(self.ls))]
            else:
                self.data = [numpy.array(hdf_file.get("{}".format(i + 1))) for i in range(len(self.ls))]
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

        # data to be calculated
        self.agent_view = None
        self.wall_view = None
        self.obs_pos = None
        self.obs_ori = None
        self.obs_angle = None
        self.others = None
        self.loc_vec = None

    def get_data(self):
        sensory = []
        loc = []
        for i in range(1, len(self.agent_data)):
            loc.append(self.get_loc_vec(i))
            sensory.append(self.craft_vector(i))
        return numpy.array(loc), numpy.array(sensory)

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

    def preprocess(self):
        new_path = self.filepath + ".npy"
        input = numpy.zeros((len(self.agent_data), 2 + self.num_bins + self.num_rays))
        start = perf_counter()
        for i in range(1, len(self.agent_data)):
            input[i] = self.craft_vector(i)
        end = perf_counter()
        elapsed = end - start
        numpy.save(new_path, input)

        print("Preprocessing took ", elapsed)
        # for i in numpy.load(new_path):
        #    print(i)

    def craft_vector(self, i):
        """calculate the vector v = (locomotion, agent_view, wall_view) from the raw data"""
        # get position of guppy 0 and convert it to sympy point
        self.obs_pos = (self.agent_data[i][0], self.agent_data[i][1])
        # get orientation vector
        self.obs_ori = (self.agent_data[i][2], self.agent_data[i][3])
        # calculate angle of guppy_orientation with respect to x_axis
        self.obs_angle = vec_to_angle(self.obs_ori[0], self.obs_ori[1])
        # calculate positions of other guppys
        self.others = [(self.data[j][i][0], self.data[j][i][1])
                       for j in range(len(self.ls)) if j != self.agent]
        self.others_angle = [vec_to_angle(self.data[j][i][2], self.data[j][i][3])
                             for j in range(len(self.ls)) if j != self.agent]
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

    def get_loc_vec(self, i):
        return numpy.array(get_locomotion_vec(self.agent_data[i - 1], self.agent_data[i]))

    def run_sim(self, step):
        for frame in range(1, len(self.agent_data), step):
            self.craft_vector(frame)
            self.dist_difference = dist_travelled(self.agent_data[frame - 1], self.agent_data[frame])
            self.plot_guppy_bins()
            # self.plot_wall_rays() #use either of the two plot_functions, not both at once

    def guppy_distances(self):
        # get the boundaries of the bins by dividing 180 degree field of view by number of bins
        # and make a ray for each resulting angle. To adjust the rays to our angle-system, we have to subtract pi / 2

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

        """
        Variant 3: Start with bins but dont delete guppys, so you can just use two for_loops
        for i in range(len(self.bins)):
            for j in range(len(self.others)):
                distance = dist(self.obs_pos, others_c[j])
                if distance < self.agent_view[i] \
                        and self.bins[i].contains(shapely.geometry.Point(others_c[j][0], others_c[j][1])):
                    self.agent_view[i] = distance
            self.agent_view[i] = intensity_linear(self.agent_view[i])
        """
        """
        Variant 2: start the loop with the guppys and break if you found its bin
        for guppy in self.others:
            for i in range(len(self.bins)):
                if self.bins[i].encloses_point(guppy):
                    distance = float(self.obs_pos.distance(guppy))
                    if distance < self.agent_view[i]:
                        self.agent_view[i] = distance
                    break
        for i in range(len(self.agent_view)):
            self.agent_view[i] = intensity_linear(self.agent_view[i], max_dist)
        """

    def plot_guppy_bins(self):
        self.ax.cla()  # clear axes

        # plot tank
        ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
        polygon = shapely.geometry.Polygon(ext)
        plot_coords(self.ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.6, zorder=1)
        self.ax.add_patch(patch)

        # plot bins
        for i, p in enumerate(self.bins):
            patch = PolygonPatch(p, facecolor=BLACK, edgecolor=YELLOW,
                                 alpha=(1 - self.agent_view[i]), zorder=1)
            self.ax.add_patch(patch)

        # plot fishes
        self.ax.plot(self.obs_pos[0], self.obs_pos[1], "go")  # self
        for i in range(len(self.others)):
            self.ax.plot([self.others[i][0]], [self.others[i][1]], "r.")  # others

        # plot legend
        self.ax.text(110, 100, 'angle: {:.2f}°'.format(degrees(self.obs_angle)))
        self.ax.text(110, 90, 'o_vector: ({:.2f},{:.2f})'.format(self.obs_ori[0], self.obs_ori[1]))
        self.ax.text(110, 80, 'angular turn: {:.10f}'.format(self.loc_vec[0]))
        self.ax.text(110, 70, 'linear speed: {:.10f}'.format(self.loc_vec[1]))
        self.ax.text(110, 60, 'dist travelled: {:.10f}'.format(self.dist_difference))

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
        self.length = len(filepaths)
        self.filepaths = filepaths
        self.data = []
        self.datapaths = []
        self.labelpaths = []

        for i in range(self.length):
            datapath = self.filepaths[i]
            datapath += "data.multi_modal.npy" if output_model == "multi_modal" else "data.fixed.npy"
            labelpath = self.filepaths[i]
            labelpath += "label.multi_modal.npy" if output_model == "multi_modal" else "label.fixed.npy"
            if not os.path.isfile(datapath):
                gc = Guppy_Calculator(self.filepaths[i], self.agent, self.num_view_bins, self.num_rays, self.livedata)
                x_loc, x_sensory = gc.get_data()
                y_loc = numpy.roll(x_loc, -1, 0)
                if output_model == ("multi_modal"):
                    for i in range(y_loc.shape[0]):
                        y_loc[i, 0] = get_bin(y_loc[i, 0], angle_min, angle_max, num_angle_bins)
                        y_loc[i, 1] = get_bin(y_loc[i, 1], speed_min, speed_max, num_speed_bins)

                numpy.save(datapath, numpy.concatenate((x_loc[:-1, :], x_sensory[:-1, :]), 1))
                numpy.save(labelpath, y_loc[:-1, :])
            self.datapaths.append(datapath)
            self.labelpaths.append(labelpath)

        #    self.data.append((numpy.concatenate((x_loc[:-1, :], x_sensory[:-1, :]), 1), y_loc[:-1, :]))


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        data = numpy.load(self.datapaths[i])
        labels = numpy.load(self.labelpaths[i])
        return data, labels

        # return self.data[i]


if __name__ == "__main__":
    filepath = "guppy_data/couzin_torus/train/8_0002.hdf5"
    filepathlive = "guppy_data/live_female_female/train/CameraCapture2019-06-20T15_35_23_672-sub_1.hdf5"
    gc = Guppy_Calculator(filepathlive, agent=0, num_guppy_bins=20, num_wall_rays=5, livedata=True, simulation=True)
    gc.run_sim(step=1)
