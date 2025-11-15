import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        x = torch.fft.fftn(x, dim=(-3, -2, -1))  # 3D FFT over (x, y, t)
        out = torch.zeros(B, self.out_channels, X, Y, T, dtype=torch.cfloat, device=x.device)

        # Only keep low-frequency modes
        out[:, :, :self.modes_x, :self.modes_y, :self.modes_t] = \
            self.compl_mul3d(x[:, :, :self.modes_x, :self.modes_y, :self.modes_t],
                             self.weight)

        # iFFT back to real space
        out = torch.fft.ifftn(out, dim=(-3, -2, -1))
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out.real  # real tensor for next layers (split real/imag separately later if needed)


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
        return x + self.activation(out)
        # return self.activation(out)

# ========================
# 3D FNO Model
# ========================
class FNO3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, width=32, n_layers=4,
                 modes=(12, 12, 12)):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
   


    data = np.load('spatiotemporal_fields_1cm_40nJ_total.npy',)


    input_data = data[:, :1, :, :, :]
    output_data = data[:, 1:, :, :, :]

    num_data = data.shape[0]
    n_train = int(num_data*0.9)
    n_test = num_data - n_train


   

    print(f'train : {n_train}, test : {n_test}')

    train_input_data = input_data[0:n_train]
    train_output_data = output_data[0:n_train]
    test_input_data = input_data[n_train:]
    test_output_data = output_data[n_train:]

    #complex dtype to real dtype with split real and imag, [M, 1, X, Y, T] -> [M, 2, X, Y, T]
    train_input_data = np.concatenate([train_input_data.real, train_input_data.imag], axis=1)
    train_output_data = np.concatenate([train_output_data.real, train_output_data.imag], axis=1)
    test_input_data = np.concatenate([test_input_data.real, test_input_data.imag], axis=1)
    test_output_data = np.concatenate([test_output_data.real, test_output_data.imag], axis=1)
    print(train_input_data.shape, train_output_data.shape, flush=True)
    print(test_input_data.shape, test_output_data.shape, flush=True)

    # Hyperparameters
    lr = 0.01
    batch_size = 20
    epochs = 50
    width = 16
    num_layers = 4
    mode_x, mode_y, mode_t = 16, 16, 16

    # Define a model
    model = FNO3D(
            in_channels=2,
            out_channels=2,
            width=width,
            n_layers=num_layers,
            modes=(mode_x, mode_y, mode_t),
            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Print the number of tunable parameters
    print('Number of parameters : ',flush=True)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    total_loss = 0
    total_test_loss = 0
    eps = 1e-8
    total_loss_list = []
    total_test_loss_list = []
    
    # scaler = torch.cuda.amp.GradScaler()    

    for epoch in range(epochs):
        model.train()
        for i in range(0, train_input_data.shape[0], batch_size):
            optimizer.zero_grad()
            inp = torch.tensor(train_input_data[i:i+batch_size], device=device)
            target = torch.tensor(train_output_data[i:i+batch_size], device=device)
            out = model(inp)
            loss = loss_fn(out, target) / (torch.mean(target**2) + eps)
            # with torch.cuda.amp.autocast():
            #     out = model(inp)        
            #     loss = loss_fn(out, target) / (torch.mean(target**2) + eps)
            total_loss = total_loss + loss.item()             
             
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()        
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, batch {i} training complete", flush=True)
        
        model.eval()
        with torch.no_grad():
            for i in range(0, test_input_data.shape[0], batch_size):
                test_input = torch.tensor(test_input_data[i:i+batch_size], device=device)
                test_target = torch.tensor(test_output_data[i:i+batch_size], device=device)
                test_out = model(test_input)
                test_loss = loss_fn(test_out, test_target) / (torch.mean(test_target**2) + eps)
                total_test_loss = total_test_loss + test_loss.item()
            
        print(f"Epoch {epoch}, Train Loss: {total_loss / n_train}, Test Loss: {total_test_loss / n_test}", flush=True)
        total_loss_list.append(total_loss / 900)
        total_test_loss_list.append(total_test_loss / 100)
        total_loss = 0
        total_test_loss = 0

    torch.save(model.state_dict(), f"trained_model_dataset1_{int(num_data)}_{num_layers}_{width}_{mode_x}_{lr}.pth")
    print("Model parameters are saved!")



    test_input_data = torch.tensor(test_input_data[0:2], device=device)
    pred = model(test_input_data)

    pred_np = pred.detach().cpu().numpy()
    output_np_real = test_output_data[0, 0, :, :, 128]
    pred_np_real = pred_np[0, 0, :, :, 128]
    output_np_imag = test_output_data[0, 1, :, :, 128]
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
    plt.savefig(f'pred_result_{int(num_data)}_{num_layers}_{width}_{mode_x}_{lr}.png', dpi=300)

    # iterations = range(len(total_loss_list))
    total_losses = np.array(total_loss_list)
    total_test_losses = np.array(total_test_loss_list)

    np.save(f'training_loss_{int(num_data)}_{num_layers}_{width}_{mode_x}_{lr}.npy', total_losses)
    np.save(f'test_loss_{int(num_data)}_{num_layers}_{width}_{mode_x}_{lr}.npy', total_test_losses)
    # plt.figure
    # plt.plot(iterations, total_losses)
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.savefig('loss_vs_iterations.png', dpi=300)

    # plt.show()
