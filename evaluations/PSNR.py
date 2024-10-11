from PIL import Image

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0
        self.save_dir_gt = r"C:\Users\zohaib\Desktop\RAD-NeRF\trial_obama_torso\imgs\GT"
        if self.save_dir_gt:
            os.makedirs(self.save_dir_gt, exist_ok=True)
            
        self.save_dir_gen = r"C:\Users\zohaib\Desktop\RAD-NeRF\trial_obama_torso\imgs\Gen"
        if self.save_dir_gen:
            os.makedirs(self.save_dir_gen, exist_ok=True)
            

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]

        # Save the images if save_dir_gt is specified
        if self.save_dir_gt:
            batch_size = preds.shape[0]
            for i in range(batch_size):
                truth_img = (truths[i] * 255).astype(np.uint8)
                truth_image_path = os.path.join(self.save_dir_gt, f"{self.N}.png")
                Image.fromarray(truth_img).save(truth_image_path)


        # Save the images if save_dir_gen is specified
        if self.save_dir_gen:
            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_img = (preds[i] * 255).astype(np.uint8)   
                pred_image_path = os.path.join(self.save_dir_gen, f"{self.N}.png")
                Image.fromarray(pred_img).save(pred_image_path)
                
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'
