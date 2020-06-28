
# data
num_guppy_bins = 20
num_wall_rays = 20
input_dim = num_guppy_bins + num_wall_rays + 2
agent = 0
num_angle_bins = 10
num_speed_bins = 10  # take only 2 bins for the speed data which is constant in the simulated data
angle_min = -0.4
angle_max = 0.4
speed_min = 0.00
speed_max = 1.00
output_dim = num_angle_bins + num_speed_bins

output_model = ""

# network
num_layers = 2
hidden_layer_size = 100
batch_size = 4
