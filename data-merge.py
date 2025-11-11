import numpy as np
import os

filenames = ['spatiotemporal_fields_2.npy', 'spatiotemporal_fields_3.npy', 'spatiotemporal_fields_4.npy']
output_file = 'total_data.npy'

<<<<<<< HEAD
# 각 파일의 shape 읽기
shapes = [np.load(f, mmap_mode='r').shape for f in filenames]
dtypes = [np.load(f, mmap_mode='r').dtype for f in filenames]

# dtype 일치 검사
if len(set(dtypes)) != 1:
    raise ValueError("All input files must have the same dtype")
=======
# current directory change to the directory of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
>>>>>>> f51e0cfa1eed61fb108823269ec5c7d722726c72

# 합칠 총 길이 계산
total_len = sum(s[0] for s in shapes)
base_shape = list(shapes[0])
base_shape[0] = total_len  # axis 0 크기만 변경

<<<<<<< HEAD
merged = np.memmap(output_file, dtype=dtypes[0], mode='w+', shape=tuple(base_shape))

start = 0
for fname in filenames:
    data = np.load(fname, mmap_mode='r')
    end = start + data.shape[0]
    print(f"Copying {fname} -> [{start}:{end})")
    merged[start:end] = data
    start = end
    del data

merged.flush()
del merged
=======
data_1 = np.load('spatiotemporal_fields_2.npy')
print(f'The shape of data_1 is {data_1.shape}')
data_2 = np.load('spatiotemporal_fields_3.npy')
print(f'The shape of data_2 is {data_2.shape}')
data_3 = np.load('spatiotemporal_fields_4.npy')
print(f'The shape of data_3 is {data_3.shape}')

total_spatiotemporal_fields = np.concatenate([data_1, data_2, data_3], axis=0)
print(f'The total number of spatiotemporal fields is {total_spatiotemporal_fields.shape[0]}')

np.save('spatiotemporal_fields_1cm.npy', total_spatiotemporal_fields)
>>>>>>> f51e0cfa1eed61fb108823269ec5c7d722726c72
