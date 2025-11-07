import numpy as np
import matplotlib.pyplot as plt

spatiotemporal_data = np.load('spatiotemporal_fields_2.npy')

print(spatiotemporal_data.shape)
print(spatiotemporal_data.dtype)

target_input = spatiotemporal_data[0, 0,]
target_output = spatiotemporal_data[0, 1,]
target_input2 = spatiotemporal_data[1, 0,]
target_output2 = spatiotemporal_data[1, 1,]

# use only center of temporal profile

# integrate spatial domain to get temporal profile
input_temporal_profile = np.sum(np.abs(target_input)**2, axis=(0,1))
output_temporal_profile = np.sum(np.abs(target_output)**2, axis=(0,1))
input_temporal_profile2 = np.sum(np.abs(target_input2)**2, axis=(0,1))
output_temporal_profile2 = np.sum(np.abs(target_output2)**2, axis=(0,1))
# plot temporal profile
plt.figure(figsize=(10, 5))
plt.plot(input_temporal_profile, label='input')
plt.plot(output_temporal_profile, label='output')
plt.plot(input_temporal_profile2, label='input2')
plt.plot(output_temporal_profile2, label='output2')
plt.legend()
plt.show()