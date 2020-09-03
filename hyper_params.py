from math import pi
# data
include_others_ori = True
live_data = False
agent_view_field = round(pi + pi / 2, 2)
far_plane = 140
num_guppy_bins = 60
num_wall_rays = 60
input_dim = num_guppy_bins + num_wall_rays + 2
input_dim += num_guppy_bins if include_others_ori else 0
agent = 0
# for multimodal
num_angle_bins = 160
num_speed_bins = 80
angle_min = -0.1
angle_max = 0.1
speed_min = -0.4
speed_max = 2.8
output_dim = num_angle_bins + num_speed_bins
dropout = 0.1
# network
output_model = "fixed"
#output_model = "multi_modal"
arch = ""

if arch == "ey":
    num_layers = 3
    hidden_layer_size = 100
    batch_size = 16
else:
    num_layers = 2
    hidden_layer_size = 300
    batch_size = 16

valbatch_size = 2
overall_model = output_model + arch
network_path = "saved_networks/guppy_net_{}_{}_hidden{}_layers{}_gbins{}_wbins{}_far_plane{}.pth".format("live" if live_data else "sim",
                                                              output_model, hidden_layer_size, num_layers,
                                                                              num_guppy_bins, num_wall_rays, far_plane)
hyperparams = {"include_others_ori": include_others_ori,
               "live_data" : live_data,
               "agent_view_field": agent_view_field,
               "far_plane": far_plane,
               "num_guppy_bins": num_guppy_bins,
               "num_wall_rays": num_wall_rays,
               "input_dim": input_dim,
               "num_angle_bins": num_angle_bins,
               "num_speed_bins": num_speed_bins,
               "angle_min": angle_min,
               "angle_max": angle_max,
               "speed_min": speed_min,
               "speed_max": speed_max,
               "output_dim": output_dim,
               "output_model": output_model,
               "arch": arch,
               "num_layers": num_layers,
               "hidden_layer_size": hidden_layer_size,
               "batch_size": batch_size,
               "overall_model": overall_model,
               }


