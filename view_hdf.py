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


# convert a unit-vector giving the orientation to its angle to the x-axis (in radians)
def vec_to_angle(x, y):
    if y >= 0:
        return acos(x)
    else:
        return 2 * pi - acos(x)


# map distance to value in range(0,1) - greater distance -> lesser intensity
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


def plot_wall_rays(tank, obs_pos: sympy.Point2D, intersections):
    #    fig = pyplot.figure(1, figsize=SIZE, dpi=90)
    #    ax = fig.add_subplot(121)

    # set up axes
    fig, ax = pyplot.subplots()
    set_limits(ax, 0, 100, 0, 100)
    ax.set_title('tank')

    # plot tank
    ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
    # int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
    polygon = ShapelyPolygon(ext)
    # plot_coords(ax, polygon.interiors[0])
    plot_coords(ax, polygon.exterior)
    patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.5, zorder=1)
    ax.add_patch(patch)

    # plot rays
    ax.plot(obs_pos.x, obs_pos.y, "ro")
    for (x, y) in intersections:
        ax.plot([obs_pos.x, x], [obs_pos.y, y])

    pyplot.show()


def plot_guppy_bins(tank, obs_pos: sympy.Point2D, bins, bin_intensities, others, intersections):
    #    fig = pyplot.figure(1, figsize=SIZE, dpi=90)
    #    ax = fig.add_subplot(121)

    # set up axes
    fig, ax = pyplot.subplots()
    ax.set_title('tank')
    set_limits(ax, 0, 100, 0, 100)

    # plot tank
    ext = [(0, 0), (0, 100), (100, 100), (100, 0)]
    polygon = shapely.geometry.Polygon(ext)
    plot_coords(ax, polygon.exterior)
    # int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
    # plot_coords(ax, polygon.interiors[0])
    patch = PolygonPatch(polygon, facecolor=BLUE, edgecolor=BLACK, alpha=0.6, zorder=1)
    ax.add_patch(patch)

    # plot bins
    for i, p in enumerate(bins):
        patch = PolygonPatch(sympyToShapelyPolygon(p), facecolor=BLACK, edgecolor=YELLOW,
                             alpha=(1 - bin_intensities[i]), zorder=1)
        ax.add_patch(patch)

    # plot fishes
    ax.plot(obs_pos.x, obs_pos.y, "go")
    for i in range(len(others)):
        ax.plot([others[i].x], [others[i].y], "r.")

    pyplot.show()


max_dist = 70


def wall_distances(tank: Polygon, obs_pos: sympy.Point2D, num_wall_angles: int):
    def wall_distance(angle):
        # find distance to wall along a given angle
        ray = sympy.Ray(obs_pos, angle=angle)
        intersection = ray.intersection(tank)
        return obs_pos.distance(intersection[0]), (intersection[0].x, intersection[0].y)

    # find distances for all angles
    angles = [pi * (i / num_wall_angles) + obs_angle for i in range(0, num_wall_angles + 1)]
    distances = []
    intersections = []
    for a in angles:
        d, i = wall_distance(a)
        # map distances to intensitys
        distances.append(intensity_linear(d, max_dist))
        intersections.append(i)
    return distances, intersections


def guppy_distances(tank: Polygon, obs_pos: sympy.Point2D, obs_angle: float, num_bins: int, others: [sympy.Point2D]):
    # get the boundaries of the bins by dividing 180 degree field of view by number of bins
    # and make a ray for each resulting angle
    rays = [sympy.Ray(obs_pos, angle=pi * x / num_bins + obs_angle) for x in range(0, num_bins + 1)]

    # get the intersection points of the rays with the tank walls
    intersections = [ray.intersection(tank)[0] for ray in rays]

    # model the bins as polygons with intersections points and observer position as vertices
    bins = []
    for i in range(len(intersections) - 1):
        if intersections[i][0] == intersections[i + 1][0] or intersections[i][1] == intersections[i + 1][1]:
            bins.append(Polygon(obs_pos, intersections[i], intersections[i + 1]))
        # if the intersection points overlap a corner of the tank, we have to add that corner to the polygon
        else:
            corner = addCorner(intersections[i], intersections[i + 1])
            bins.append(Polygon(obs_pos, intersections[i], corner, intersections[i + 1]))

    # loop through the bins and find the closest guppy for each bin
    bin_distances = [1000000 for i in range(len(bins))]
    length = len(others)
    others_c = others[:]
    for i in range(len(bins)):
        j = 0
        while j < length:
            if bins[i].encloses_point(others_c[j]) and obs_pos.distance(others_c[j]) < bin_distances[i]:
                bin_distances[i] = float(obs_pos.distance(others_c[j]))
                del (others_c[j])  # if guppy is already in a bin, delete it
                length -= 1
            else:
                j += 1

    print("closest distance for each bin:", bin_distances)

    # convert distances to intensities in range(0,1)
    for i in range(len(bin_distances)):
        bin_distances[i] = intensity_linear(bin_distances[i], max_dist)  # sqrt(100*100 + 100*100))

    # plot the result
    plot_guppy_bins(tank, obs_pos, bins, bin_distances, others, intersections)

    return bin_distances


with h5py.File("guppy_data/couzin_torus/train/8_0000.hdf5", "r") as f:
    ls = list(f.keys())
    print("Datasets in the first file with 8 guppys: ", ls)
    data = [f.get("{}".format(i)) for i in range(len(ls))]

    # choose guppy null
    guppy_zero_data = numpy.array(data[0])
    print(guppy_zero_data[0])

    # tank coordinates are just 100 x 100 square
    coords = [(0, 0), (100, 0), (100, 100), (0, 100)]
    tank = sympy.geometry.polygon.Polygon(*coords)

    for testframe in range(0, 700, 70):
        # get position of guppy 0 and convert it to sympy point
        obs_pos = sympy.Point2D(guppy_zero_data[testframe][0], guppy_zero_data[testframe][1])

        # get orientation vector
        obs_ori = (guppy_zero_data[testframe][2], guppy_zero_data[testframe][3])

        # calculate angle of guppy_orientation with respect to x_axis
        obs_angle = vec_to_angle(obs_ori[0], obs_ori[1])

        # calculate some angles for the rays -> divide 180 degrees into sectors
        others = [sympy.Point2D(data[i][testframe][0], data[i][testframe][1]) for i in range(1, len(ls))]

        # calculate intensity vectors for walls (wall_view)
        num_wall_rays = 4  # = length of wall_view
        wall_view, intersections = wall_distances(tank, obs_pos, num_wall_rays)
        # uncomment to plot the wall rays
        # plot_wall_rays(tank, obs_pos, intersections)

        # calculate intensity vector for nearby guppies
        num_bins = 6  # = length of agent_view
        agent_view = guppy_distances(tank, obs_pos, obs_angle, 6, others)

    # for x in intersections:
    #     print(x.evalf())
