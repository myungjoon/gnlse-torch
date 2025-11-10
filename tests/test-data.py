import numpy as np
import matplotlib.pyplot as plt

spatiotemporal_data = np.load('spatiotemporal_fields_1cm.npy')

print(f'The shape of spatiotemporal_data is {spatiotemporal_data.shape}')
print(f'The dtype of spatiotemporal_data is {spatiotemporal_data.dtype}')

target_input = spatiotemporal_data[0, 0,]
target_output = spatiotemporal_data[0, 1,]

target_input2 = spatiotemporal_data[1, 0,]
target_output2 = spatiotemporal_data[1, 1,]

target_input3 = spatiotemporal_data[2, 0,]
target_output3 = spatiotemporal_data[2, 1,]

target_input4 = spatiotemporal_data[3, 0,]
target_output4 = spatiotemporal_data[3, 1,]
# use only center of temporal profile

# integrate spatial domain to get temporal profile
input_temporal_profile = np.sum(np.abs(target_input)**2, axis=(0,1))
output_temporal_profile = np.sum(np.abs(target_output)**2, axis=(0,1))

input_temporal_profile2 = np.sum(np.abs(target_input2)**2, axis=(0,1))
output_temporal_profile2 = np.sum(np.abs(target_output2)**2, axis=(0,1))

input_temporal_profile3 = np.sum(np.abs(target_input3)**2, axis=(0,1))
output_temporal_profile3 = np.sum(np.abs(target_output3)**2, axis=(0,1))

input_temporal_profile4 = np.sum(np.abs(target_input4)**2, axis=(0,1))
output_temporal_profile4 = np.sum(np.abs(target_output4)**2, axis=(0,1))
# plot temporal profile
plt.figure(figsize=(10, 5))
plt.plot(input_temporal_profile, label='input1', color='red')
plt.plot(output_temporal_profile, label='output1', color='red', linestyle='--')
plt.plot(input_temporal_profile2, label='input2', color='blue')
plt.plot(output_temporal_profile2, label='output2', color='blue', linestyle='--')
plt.plot(input_temporal_profile3, label='input3', color='green')
plt.plot(output_temporal_profile3, label='output3', color='green', linestyle='--')
plt.plot(input_temporal_profile4, label='input4', color='purple')
plt.plot(output_temporal_profile4, label='output4', color='purple', linestyle='--')
plt.legend()
plt.show()