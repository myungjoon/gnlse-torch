

import numpy as np


import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    data = np.load('spatiotemporal_fields_5cm_20nJ_40_20.npy',)
    
    
    Nt = 2**10 // 4
    time_window = 4 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    print(f'data.shape: {data.shape}', flush=True)
    ind = 4
    input_data = data[ind, 0, :, :, 128]
    output_data = data[ind, 1, :, :, 128]

    print(f'input_data.shape: {input_data.shape}, output_data.shape: {output_data.shape}', flush=True)
    print(f'input_data.dtype: {input_data.dtype}, output_data.dtype: {output_data.dtype}', flush=True)

    input_intensity = np.abs(input_data)**2
    output_intensity = np.abs(output_data)**2

    
    input_time_data = data[ind, 0, 32, 32, :]
    output_time_data = data[ind, 1, 32, 32, :]
    print(f'input_time_data.shape: {input_time_data.shape}, output_time_data.shape: {output_time_data.shape}', flush=True)
    print(f'input_time_data.dtype: {input_time_data.dtype}, output_time_data.dtype: {output_time_data.dtype}', flush=True)



    input_fft  = np.fft.fftshift(
                    np.fft.fft(
                        np.fft.ifftshift(input_time_data,), 
                    ),
                )
    output_fft = np.fft.fftshift(
                    np.fft.fft(
                        np.fft.ifftshift(output_time_data,), 
                    ),
                )

    input_spectral_intensity = np.abs(input_fft)**2
    output_spectral_intensity = np.abs(output_fft)**2

    input_time_intensity = np.abs(input_time_data)**2
    output_time_intensity = np.abs(output_time_data)**2

    dx = 4 * 62.5 / 2 / 128 * 1e-6
    dy = 4 *62.5 / 2 / 128 * 1e-6
    dt = 3 / 2**9 * 1e-12
    # check the integral (pulse energy)
    # for ind in range(data.shape[0]):
    #     input_energy = np.sum(np.abs(data[ind, 0])**2) * dx * dy * dt * 1e9
    #     output_energy = np.sum(np.abs(data[ind, 1])**2) * dx * dy * dt * 1e9
    #     print(f'input_energy: {input_energy}, output_energy: {output_energy}', flush=True)

    # plot_intensities(input_data, output_data)
    


    # 1. 전체 필드 가져오기 (x, y 전체)
    # shape: (Nx, Ny, Nt) 라고 가정
    input_field_3d  = data[ind, 0, :, :, :] 
    output_field_3d = data[ind, 1, :, :, :]

    print(f'Input Field Shape: {input_field_3d.shape}', flush=True) 

    # ==========================================
    # A. 시간 반응 (Temporal Response) - 오실로스코프/Power Meter 모사
    # ==========================================
    # 1. 각 지점의 Intensity 계산 (|E|^2)
    input_intensity_3d  = np.abs(input_field_3d)**2
    output_intensity_3d = np.abs(output_field_3d)**2

    # 2. 공간 축(x, y)에 대해 합산 (Sum) -> 결과: (Nt,) 1D array
    input_total_power  = np.sum(input_intensity_3d, axis=(0, 1))
    output_total_power = np.sum(output_intensity_3d, axis=(0, 1))


    # ==========================================
    # B. 스펙트럼 반응 (Spectral Response) - OSA 모사
    # ==========================================
    # 1. 시간 축(axis=-1)에 대해서만 FFT 수행
    # 주의: fftshift, ifftshift에도 axes=-1을 줘야 공간축이 뒤섞이지 않음
    input_fft_3d = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.ifftshift(input_field_3d, axes=-1), 
                            axis=-1
                        ), 
                        axes=-1
                    )

    output_fft_3d = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.ifftshift(output_field_3d, axes=-1), 
                            axis=-1
                        ), 
                        axes=-1
                    )

    # 2. 각 지점의 Spectral Intensity 계산
    input_spectral_intensity_3d  = np.abs(input_fft_3d)**2
    output_spectral_intensity_3d = np.abs(output_fft_3d)**2

    # 3. 공간 축(x, y)에 대해 합산 (Sum) -> 결과: (Nt,) 1D array
    input_total_spectrum  = np.sum(input_spectral_intensity_3d, axis=(0, 1))
    output_total_spectrum = np.sum(output_spectral_intensity_3d, axis=(0, 1))

    #subplots for input and output intensity
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(input_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(output_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()

    plt.figure()
    plt.plot(t, input_time_intensity, label='input')
    plt.plot(t, output_time_intensity, label='output')
    plt.legend()

    plt.figure()
    plt.plot(input_spectral_intensity, label='input')
    plt.plot(output_spectral_intensity, label='output')
    plt.legend()


    plt.figure()
    plt.title('Total Power')
    plt.plot(input_total_power, label='input')
    plt.plot(output_total_power, label='output')
    plt.legend()

    plt.figure()
    plt.title('Total Spectral Intensity')
    plt.plot(input_total_spectrum, label='input')
    plt.plot(output_total_spectrum, label='output')
    plt.legend()

    plt.show()