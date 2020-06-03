import h5py
import numpy
import sympy
from sympy.geometry.polygon import Polygon
from math import *
from matplotlib import pyplot
from shapely.geometry import Polygon as ShapelyPolygon
import shapely
from descartes.patch import PolygonPatch
from figures import BLUE, RED, BLACK, YELLOW, SIZE, set_limits, plot_coords, color_isvalid


# Whole code is inspired by Moritz Maxeiners master thesis and his code for the thesis


# some simple helper functions

# convert a unit-vector giving the orientation to its angle to the x-axis (in radians)
def vec_to_angle(x, y):
    if y >= 0:
        return acos(x)
    else:

        return -acos(x)


max_dist = 70


# map distance to value in range(0,1) - greater distance -> lesser intensity (from Moritz Maxeiner)
def intensity_linear(dist, max_dist):
    if dist < 0 or max_dist < dist:
        return 0
    return 1 - float(dist) / max_dist


def sympyToShapelyPolygon(p: Polygon):
    vertices = []
    for v in p.vertices:
        vertices.append((v.x, v.y))
    return shapely.geometry.Polygon(vertices)


def addCorner(p1, p2):
    x = p2.x if 0 < p1.x < 100 else p1.x
    y = p2.y if 0 < p1.y < 100 else p1.y
    return sympy.Point2D(x, y)


def dot_product(vec1, vec2):
    akk = 0
    for i in range(len(vec1)):
        akk += vec1[i] * vec2[i]
    return akk


def get_locomotion_vec(data_prev, data_cur):
    if data_prev is None:
        return 0, 0
    else:
        ang_turn = vec_to_angle(data_prev[2], data_prev[3]) - vec_to_angle(data_cur[2], data_cur[3])
        diff_vec = ((data_cur[0] - data_prev[0]), (data_cur[1] - data_prev[1]))
        linear_speed = dot_product(diff_vec, (data_cur[2], data_cur[3]))
        return ang_turn, linear_speed


def dist_travelled(data_prev, data_cur):
    if data_prev is None:
        return 0, 0
    else:
        return sqrt((data_cur[0] - data_prev[0]) ** 2 + (data_cur[1] - data_prev[1]) ** 2)


# thats the main class, its run method will do all the work
class Guppy_Calculator():
    def __init__(self, hdf_file, agent, num_bins, num_rays):
        # set up data
        self.ls = list(hdf_file.keys())
        self.data = [hdf_file.get("{}".format(i)) for i in range(len(self.ls))]
        self.agent = agent
        self.agent_data = numpy.array(self.data[agent])

        # set up axes
        self.fig, self.ax = pyplot.subplots()
        self.ax.set_title('tank')
        set_limits(self.ax, 0, 100, 0, 100)
        # tank coordinates are just 100 x 100 square
        coords = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.tank = sympy.geometry.polygon.Polygon(*coords)

        # data to be calculated
        self.num_bins = num_bins
        self.num_rays = num_rays
        self.bin_angles = [pi * (i / self.num_bins) - pi / 2 for i in range(0, self.num_bins + 1)]
        self.wall_angles = [pi * (i / self.num_rays) - pi / 2 for i in range(0, self.num_rays)]
        self.agent_view = None
        self.wall_view = None
        self.obs_pos = None
        self.obs_ori = None
        self.obs_angle = None
        self.others = None
        self.loc_vec = None

    def craft_vector(self, i):
        """calculate the vector v = (locomotion, agent_view, wall_view) from the raw data"""
        # get position of guppy 0 and convert it to sympy point
        self.obs_pos = sympy.Point2D(self.agent_data[i][0], self.agent_data[i][1])
        # get orientation vector
        self.obs_ori = (self.agent_data[i][2], self.agent_data[i][3])
        # calculate angle of guppy_orientation with respect to x_axis
        self.obs_angle = vec_to_angle(self.obs_ori[0], self.obs_ori[1])
        # calculate positions of other guppys
        self.others = [sympy.Point2D(self.data[j][i][0], self.data[j][i][1]) for j in
                       range(len(self.ls)) if j != self.agent]
        # calculate intensity vector wall_view for distance to walls
        self.wall_distances()
        # calculate intensity vector agent_view for distance to nearby guppies
        self.guppy_distances()
        # loc_vec[0] = angular turn; loc_vec[1] = linear speed
        self.loc_vec = get_locomotion_vec(self.agent_data[i - 1], self.agent_data[i])
        # we return the vector already concatenated in a numpy_vector
        return numpy.concatenate((numpy.array(self.loc_vec), numpy.array(self.agent_view), numpy.array(self.wall_view)))

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
        agent_rays = [sympy.Ray(self.obs_pos, angle=x + self.obs_angle) for x in self.bin_angles]
        intersections = [ray.intersection(self.tank)[0] for ray in agent_rays]

        # construct the bins as polygons with intersection points and observer position as vertices
        # would surely be more efficient to just use a function which checks if a point lies within a polygon defined
        # by the three / four points given.
        self.bins = []
        for i in range(len(intersections) - 1):
            if intersections[i][0] == intersections[i + 1][0] or intersections[i][1] == intersections[i + 1][1]:
                self.bins.append(Polygon(self.obs_pos, intersections[i], intersections[i + 1]))
            else:  # if the intersection points overlap a corner of the tank, we have to add that corner to the polygon
                corner = addCorner(intersections[i], intersections[i + 1])
                self.bins.append(Polygon(self.obs_pos, intersections[i], corner, intersections[i + 1]))

        # loop through the bins and find the closest guppy for each bin
        self.agent_view = [1000.0 for i in range(len(self.bins))]
        length = len(self.others)
        others_c = self.others[:]

        # Variant 1: Start with the bins and delete guppys which already found their bin
        # This seems to be the most efficient
        for i in range(len(self.bins)):
            j = 0
            while j < length:
                if self.bins[i].encloses_point(others_c[j]):
                    distance = float(self.obs_pos.distance(others_c[j]))
                    if distance < self.agent_view[i]:
                        self.agent_view[i] = distance
                    del (others_c[j])
                    length -= 1
                else:
                    j += 1

            self.agent_view[i] = intensity_linear(self.agent_view[i], max_dist)

        # agent_view vector is ready now
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
        """
        Variant 3: Start with bins but dont delete guppys, so you can just use two for_loops
        for i in range(len(self.bins)):
            for j in range(len(self.others)):
                distance = float(self.obs_pos.distance(others_c[j]))
                if distance < self.agent_view[i] and self.bins[i].encloses_point(others_c[j]):
                    self.agent_view[i] = distance
            
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
            patch = PolygonPatch(sympyToShapelyPolygon(p), facecolor=BLACK, edgecolor=YELLOW,
                                 alpha=(1 - self.agent_view[i]), zorder=1)
            self.ax.add_patch(patch)

        # plot fishes
        self.ax.plot(self.obs_pos.x, self.obs_pos.y, "go")  # self
        for i in range(len(self.others)):
            self.ax.plot([self.others[i].x], [self.others[i].y], "r.")  # others

        # plot legend
        self.ax.text(110, 100, 'angle: {:.2f}°'.format(degrees(self.obs_angle)))
        self.ax.text(110, 90, 'o_vector: ({:.2f},{:.2f})'.format(self.obs_ori[0], self.obs_ori[1]))
        self.ax.text(110, 80, 'angular turn: {:.10f}'.format(self.loc_vec[0]))
        self.ax.text(110, 70, 'linear speed: {:.10f}'.format(self.loc_vec[1]))
        self.ax.text(110, 60, 'dist travelled: {:.10f}'.format(self.dist_difference))

        pyplot.show(block=False)
        pyplot.pause(0.00000000000001)

    def wall_distances(self):
        self.wall_rays = [sympy.Ray(self.obs_pos, angle=x + self.obs_angle) for x in self.wall_angles]
        self.intersections = []
        self.wall_view = []
        for ray in self.wall_rays:
            intersection = ray.intersection(self.tank)[0]
            self.intersections.append(intersection)
            self.wall_view.append(intensity_linear(self.obs_pos.distance(intersection), max_dist))

    def plot_wall_rays(self):
        self.ax.cla()
        # plot tank
        ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
        polygon = shapely.geometry.Polygon(ext)
        plot_coords(self.ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.6, zorder=1)
        self.ax.add_patch(patch)

        # plot rays
        self.ax.plot(self.obs_pos.x, self.obs_pos.y, "ro")
        for (x, y) in self.intersections:
            self.ax.plot([self.obs_pos.x, x], [self.obs_pos.y, y])

        pyplot.show(block=False)
        pyplot.pause(0.00000000001)


if __name__ == "__main__":
    with h5py.File("guppy_data/couzin_torus/train/8_0000.hdf5", "r") as f:
        # agent sets the guppy considered as agent, num_bins defines the view field division, num_rays the wall_rays
        gc = Guppy_Calculator(f, agent=0, num_bins=7, num_rays=5)
        print("Example: Handcrafted vector for frame 100:\n", gc.craft_vector(100))
        gc.run_sim(step=10)
