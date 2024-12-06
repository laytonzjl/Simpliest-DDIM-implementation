import os
import torch
import model
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn as nn

# 初始化参数
t = 1000
channel = 64
channel_multi = [1, 2, 2, 2]
if_attention = [1]
num_resblock = 3
grad_clip = 1
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
sample_interval = 30


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, unet, beta_start, beta_end, timesteps, device):
        super().__init__()
        self.unet = unet
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.device = device

        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(device)
        self.alpha_schedule = 1. - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)

    def sample(self, x_T):
        x = x_T
        all_images = []

        for t in reversed(range(self.timesteps)):

            t_tensor = torch.tensor([t]).to(self.device)
            alpha_t = self.alpha_schedule[t]
            alpha_t_cumprod = self.alpha_cumprod[t]
            beta_t = self.beta_schedule[t]
            pred_noise = self.unet(x, t_tensor)

            if t > 0:
                x = (1. / torch.sqrt(alpha_t)) * (x - (1. - alpha_t) / torch.sqrt(1. - alpha_t_cumprod) * pred_noise) + torch.sqrt(beta_t) * torch.randn_like(x) * 0.6
                x = torch.clamp(x, min=-1, max=2)
            else:
                x = (1. / torch.sqrt(alpha_t)) * (x - (1. - alpha_t) / torch.sqrt(1. - alpha_t_cumprod) * pred_noise)
                x = torch.clamp(x, min=-1, max=2)

            if t % sample_interval == 0:
                all_images.append(x.cpu())

            torch.cuda.empty_cache()

        return x, all_images


unet = model.UNet(t, channel, channel_multi, if_attention, num_resblock, dropout).to(device)
unet.load_state_dict(torch.load("middle_ema_model.pth"))
unet.eval()
sampler = GaussianDiffusionSampler(unet, 1e-4, 2e-2, 300, device)

x_T = torch.randn(1, 3, 32, 32).to(device).float()
output_dir = "sampler"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    generated_image, all_images = sampler.sample(x_T)

vutils.save_image(generated_image, os.path.join(output_dir, "generated.png"))
num_steps = len(all_images)
fig, axes = plt.subplots(1, num_steps, figsize=(20, 20))

for i, img in enumerate(all_images):
    axes[i].imshow(img[0].permute(1, 2, 0).cpu().numpy())
    axes[i].set_title(f"sample_step:{(i+1)*sample_interval}")
    axes[i].axis('off')
plt.show()

for i, img in enumerate(all_images):
    vutils.save_image(img, os.path.join(output_dir, f"sample_step _{(i+1)*sample_interval}.png"))

