import numpy as np

filenames = ['spatiotemporal_fields_2.npy', 'spatiotemporal_fields_3.npy', 'spatiotemporal_fields_4.npy']
output_file = 'total_data.npy'

# 각 파일의 shape 읽기
shapes = [np.load(f, mmap_mode='r').shape for f in filenames]
dtypes = [np.load(f, mmap_mode='r').dtype for f in filenames]

# dtype 일치 검사
if len(set(dtypes)) != 1:
    raise ValueError("All input files must have the same dtype")

# 합칠 총 길이 계산
total_len = sum(s[0] for s in shapes)
base_shape = list(shapes[0])
base_shape[0] = total_len  # axis 0 크기만 변경

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
