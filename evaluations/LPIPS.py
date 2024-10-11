import os
from PIL import Image
import numpy as np
import torch
import lpips  # Ensure lpips is installed
from torchvision.models import AlexNet_Weights

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


def compute_mean_lpips(pred_folder, gt_folder, net='alex'):
    lpips_meter = LPIPSMeter(net=net)

    pred_images = sorted(os.listdir(pred_folder))
    gt_images = sorted(os.listdir(gt_folder))

    for pred_image_name, gt_image_name in zip(pred_images, gt_images):
        pred_image_path = os.path.join(pred_folder, pred_image_name)
        gt_image_path = os.path.join(gt_folder, gt_image_name)

        pred_image = np.array(Image.open(pred_image_path).convert('RGB')) / 255.0
        gt_image = np.array(Image.open(gt_image_path).convert('RGB')) / 255.0

        pred_tensor = torch.tensor(pred_image).float().unsqueeze(0)  # Add batch dimension
        gt_tensor = torch.tensor(gt_image).float().unsqueeze(0)      # Add batch dimension

        lpips_meter.update(pred_tensor, gt_tensor)

    mean_lpips = lpips_meter.measure()
    return mean_lpips

# Example usage
pred_folder = r"/home/host/pegah/evl/RAD-NeRF/hu/donya/GH"
gt_folder = r"/home/host/pegah/evl/RAD-NeRF/hu/donya/GT"
mean_lpips = compute_mean_lpips(pred_folder, gt_folder, net='alex')
print(f'Mean LPIPS: {mean_lpips:.6f}')
