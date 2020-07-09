from math import pi
# data
include_others_angles = True
live_data = True

agent_view_field = pi + pi / 2
far_plane = 140
num_guppy_bins = 20
num_wall_rays = 20
input_dim = num_guppy_bins + num_wall_rays + 2
input_dim += num_guppy_bins if include_others_angles else 0
agent = 0


# for multimodal
num_angle_bins = 40
num_speed_bins = 40  
angle_min = -2
angle_max = 2
speed_min = -0.8
speed_max = 2.8
output_dim = num_angle_bins + num_speed_bins


# network
output_model = "multi_modal"
#output_model = "fixed"
num_layers = 1
hidden_layer_size = 400
batch_size = 4
network_path = "guppy_net_{}_{}_hidden{}_layers{}_gbins{}_wbins{}.pth".format("live" if live_data else "sim",
                                                              output_model, hidden_layer_size, num_layers,
                                                                              num_guppy_bins, num_wall_rays)
