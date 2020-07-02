from math import pi
# data
include_others_angles = True
live_data = True
num_guppy_bins = 20
num_wall_rays = 20
input_dim = num_guppy_bins + num_wall_rays + 2
input_dim += num_guppy_bins if include_others_angles else 0


# for multimodal
agent = 0
num_angle_bins = 40
num_speed_bins = 40  # take only 2 bins for the speed data which is constant in the simulated data
angle_min = -2
angle_max = 2
speed_min = -0.8
speed_max = 2.8
output_dim = num_angle_bins + num_speed_bins


# network
output_model = "multi_modal"
#output_model = "fixed"
num_layers = 2
hidden_layer_size = 200
batch_size = 4
