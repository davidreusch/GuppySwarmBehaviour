from math import pi
# data
include_others_angles = True
live_data = False

agent_view_field = round(pi + pi / 2, 2)
far_plane = 140
num_guppy_bins = 60
num_wall_rays = 60
input_dim = num_guppy_bins + num_wall_rays + 2
input_dim += num_guppy_bins if include_others_angles else 0
agent = 0


# for multimodal
num_angle_bins = 160
num_speed_bins = 80
angle_min = -0.1
angle_max = 0.1
speed_min = -0.8
speed_max = 2.8
output_dim = num_angle_bins + num_speed_bins


# network
output_model = "fixed"
#output_model = "multi_modal"
arch = ""

if arch == "ey":
    num_layers = 3
    hidden_layer_size = 100
    batch_size = 4
else:
    num_layers = 1
    hidden_layer_size = 400
    batch_size = 4

network_path = "saved_networks/guppy_net_{}_{}_hidden{}_layers{}_gbins{}_wbins{}_far_plane{}.pth".format("live" if live_data else "sim",
                                                              output_model, hidden_layer_size, num_layers,
                                                                              num_guppy_bins, num_wall_rays, far_plane)
