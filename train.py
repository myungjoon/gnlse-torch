import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn
import torch.fft

import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========================
# 3D Fourier Layer
# ========================
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # modes = (modes_x, modes_y, modes_t)
        self.modes_x, self.modes_y, self.modes_t = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels,
                                                       self.modes_x, self.modes_y, self.modes_t, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def compl_mul3d(self, input, weights):
        # input: (B, in_c, kx, ky, kt)
        # weights: (in_c, out_c, kx, ky, kt)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B, C, X, Y, T = x.shape
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))  # 3D FFT over (x, y, t)
        out_ft = torch.zeros(B, self.out_channels, X, Y, T, dtype=torch.cfloat, device=x.device)

        # Only keep low-frequency modes
        out_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_t] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_t],
                             self.weight)

        # iFFT back to real space
        x_out = torch.fft.ifftn(out_ft, dim=(-3, -2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias.view(1, -1, 1, 1, 1)
        return x_out.real  # real tensor for next layers (split real/imag separately later if needed)


# ========================
# 3D FNO Block
# ========================
class FNO3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=(12, 12, 8), width=64):
        super().__init__()
        self.spectral_conv = SpectralConv3d(in_channels, out_channels, modes)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.spectral_conv(x) + self.pointwise_conv(x)
        return self.activation(out)


# ========================
# 3D FNO Model
# ========================
class FNO3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, width=64, n_layers=4,
                 modes=(12, 12, 8)):
        super().__init__()
        self.input_proj = nn.Conv3d(in_channels, width, 1)
        self.fno_layers = nn.ModuleList([
            FNO3DBlock(width, width, modes, width) for _ in range(n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv3d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv3d(width // 2, out_channels, 1)
        )

    def forward(self, x):
        # x: (B, 2, X, Y, T)
        x = self.input_proj(x)
        for layer in self.fno_layers:
            x = layer(x)
        return self.output_proj(x)


# ========================
# Example usage
# ========================
if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = FNO3D(
        in_channels=2,        # Re/Im
        out_channels=2,       # Re/Im
        width=32,             # internal width
        n_layers=4,           # number of Fourier blocks
        modes=(12, 12, 12)     # number of Fourier modes in x, y, t
    ).to(device)

    data = np.load('spatiotemporal_fields_1000.npy')
    print(data.shape)

    input_data = data[:, :1, :, :, :]
    output_data = data[:, 1:, :, :, :]

    #complex dtype to real dtype, [M, 1, X, Y, T] -> [M, 2, X, Y, T] 1 to 2 is complex to real/imag, input_data is already complex, numpy type
    input_data = np.concatenate([input_data.real, input_data.imag], axis=1)
    output_data = np.concatenate([output_data.real, output_data.imag], axis=1)
    print(input_data.shape, output_data.shape)

    # plt.figure(figsize=(5, 5))
    # data0 = input_data[0, ..., 128]
    # data0_intensity = np.sqrt(data0[0]**2 + data0[1]**2)
    # plt.imshow(data0_intensity, cmap='turbo', vmin=0, vmax=np.max(data0_intensity))
    # plt.colorbar()
    # plt.savefig('data0_intensity.png', dpi=300)
    
    total_loss = 0


    batch_size = 2
    # optimizer : Adam, loss : relative MAE (normalized by the ground truth)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()
    epochs = 200
    total_loss_list = []
    for epoch in range(epochs):
        for i in range(0, input_data.shape[0], batch_size):
            inp = torch.tensor(input_data[i:i+batch_size], device=device)
            target = torch.tensor(output_data[i:i+batch_size], device=device)
            out = model(inp)
            loss = loss_fn(out, target) / torch.mean(torch.abs(target))
            total_loss = total_loss + loss.item()             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        print(f"Epoch {epoch}, Loss: {total_loss / 5}")
        total_loss_list.append(total_loss / 5)
        total_loss = 0
    

    input_data = torch.tensor(input_data[0:2], device=device)
    pred = model(input_data)

    pred_np = pred.detach().cpu().numpy()
    output_np_real = output_data[0, 0, :, :, 128]
    pred_np_real = pred_np[0, 0, :, :, 128]
    
    output_np_imag = output_data[0, 1, :, :, 128]
    pred_np_imag = pred_np[0, 1, :, :, 128]

    output_intensity = np.sqrt(output_np_real**2 + output_np_imag**2)
    pred_intensity = np.sqrt(pred_np_real**2 + pred_np_imag**2)




    vmin = 0
    vmax = np.max(np.concatenate([output_intensity, pred_intensity], axis=0))
    plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(output_intensity, cmap='turbo', vmin=vmin,)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(pred_intensity, cmap='turbo', vmin=vmin, )
    plt.colorbar()
    plt.savefig('out_np_real.png', dpi=300)

    # iterations = range(len(total_loss_list))
    # total_losses = np.array(total_loss_list)
    # plt.figure
    # plt.plot(iterations, total_losses)
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.savefig('loss_vs_iterations.png', dpi=300)

    plt.show()