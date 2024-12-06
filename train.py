import torch
import torch.nn as nn
import model
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from torch.nn import functional as F
import copy
import os
from PIL import Image

total_steps = 20000
batch_size = 128
lr = 2e-4
t = 1000
channel = 64
channel_multi = [1, 2, 2, 2]
if_attention = [1]
num_resblock = 3
grad_clip = 1
dropout = 0.1
dataset_path = './dataset'
log_interval = 100
save_interval = 5000
warmup = 1000
decay = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"

# 网络初始化
unet = model.UNet(t, channel, channel_multi, if_attention, num_resblock, dropout).to(device)
for p in unet.parameters():
    if p.dim() > 1:
        if isinstance(p, nn.Conv2d):
            nn.init.kaiming_normal_(p, mode='fan_in')
        else:
            nn.init.xavier_uniform_(p)
model_size = 0
for param in unet.parameters():
    model_size += param.data.nelement()
print('模型参数量: %.2f M' % (model_size / 1024 / 1024))

ema_model = copy.deepcopy(unet)
ema_model.eval()


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, warmup) / warmup


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
indices = [i for i, label in enumerate(dataset.targets) if label == 0]
subset = Subset(dataset, indices)
loader = infiniteloop(torch.utils.data.DataLoader(
    dataset=subset, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True))
'''


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # 遍历根目录，收集所有图片路径
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 确保所有图片为 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 创建数据集和数据加载器
dataset = CustomImageDataset(root_dir='dataset', transform=transform)
loader = infiniteloop(torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True))

optim = torch.optim.Adam(unet.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
AddNoise = model.AddNoise(T=t).to(device)
writer = SummaryWriter(log_dir='runs/')

step = 0
scaler = torch.cuda.amp.GradScaler()

with trange(total_steps, dynamic_ncols=True) as pbar:
    for step in pbar:
        optim.zero_grad()
        images = next(loader).to(device)

        batch_size = images.size(0)
        timestep = torch.randint(t, size=(batch_size,), device=images.device)
        noise_image, noise = AddNoise(images, timestep)
        predict_noise = unet(noise_image, timestep)
        loss = F.mse_loss(noise, predict_noise, reduction='none').mean()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        ema(unet, ema_model, decay)

        pbar.set_postfix(loss=loss.item())
        pbar.update(1)

        # 保存到 TensorBoard
        if step % log_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), step)

        # 定期保存模型
        if step % save_interval == 0 and step > 0:
            torch.save(unet.state_dict(), f'unet_model_step_{step}.pth')
            torch.save(ema_model.state_dict(), f'ema_model_step_{step}.pth')

        # 增加步数
        step += 1

torch.save(ema_model.state_dict(), 'ema_model.pth')
torch.save(unet.state_dict(), 'unet_model.pth')
writer.close()




