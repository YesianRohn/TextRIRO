import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from utils import create_model, load_state_dict
from pytorch_lightning import seed_everything
import lmdb
import six
import numpy as np
from PIL import Image
import io
import os


class LmdbWriter:
    def __init__(self, lmdb_path, map_size=1099511627776):
        os.makedirs(lmdb_path, exist_ok=True)
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.count = 0

    def write_sample(self, orig_img, label, gen_img, orig_label):
        # orig_img, gen_img: [C,H,W] torch.Tensor or np.ndarray or PIL.Image
        # label, orig_label: str or bytes
        if isinstance(orig_img, torch.Tensor):
            orig_img = orig_img.cpu().numpy()
            orig_img = np.transpose(orig_img, (1, 2, 0))
            orig_img = (orig_img * 255).astype(np.uint8)
            orig_img = Image.fromarray(orig_img)
        if isinstance(orig_img, np.ndarray):
            orig_img = Image.fromarray(orig_img)
        if not isinstance(orig_img, Image.Image):
            raise TypeError("orig_img must be a PIL.Image, np.ndarray or torch.Tensor")

        if isinstance(gen_img, torch.Tensor):
            gen_img = gen_img.cpu().numpy()
            gen_img = np.transpose(gen_img, (1, 2, 0))
            gen_img = (gen_img * 255).astype(np.uint8)
            gen_img = Image.fromarray(gen_img)
        if isinstance(gen_img, np.ndarray):
            gen_img = Image.fromarray(gen_img)
        if not isinstance(gen_img, Image.Image):
            raise TypeError("gen_img must be a PIL.Image, np.ndarray or torch.Tensor")

        if isinstance(label, bytes):
            label = label.decode('utf-8')
        if isinstance(orig_label, bytes):
            orig_label = orig_label.decode('utf-8')

        # resize the generated image back to the original aspect (scaled by text length)
        w, h = orig_img.size
        denom = max(len(orig_label), 1)
        if h > 2 * w:
            gen_img = gen_img.rotate(-90, expand=True)
            h = max(h * len(label) // denom, 1)
        else:
            w = max(w * len(label) // denom, 1)
        if gen_img.size != (w, h):
            gen_img = gen_img.resize((w, h), Image.BICUBIC)

        orig_buf = io.BytesIO()
        orig_img.save(orig_buf, format='PNG')
        gen_buf = io.BytesIO()
        gen_img.save(gen_buf, format='PNG')

        idx = self.count + 1  # keys start from 1
        self.txn.put(f'ori-{idx:09d}'.encode(), orig_buf.getvalue())
        self.txn.put(f'label-{idx:09d}'.encode(), label.encode('utf-8'))
        self.txn.put(f'orilabel-{idx:09d}'.encode(), orig_label.encode('utf-8'))
        self.txn.put(f'image-{idx:09d}'.encode(), gen_buf.getvalue())
        self.txn.put(f'wh-{idx:09d}'.encode(), f'{w}_{h}'.encode())
        self.count += 1

        # commit every 1000 samples
        if self.count % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.put('num-samples'.encode(), str(self.count).encode())
        self.txn.commit()
        self.env.close()


class LmdbTextDataset(Dataset):
    def __init__(self, lmdb_paths, size=(64, 256)):
        super().__init__()
        if isinstance(lmdb_paths, str):
            lmdb_paths = [lmdb_paths]
        self.envs = []
        self.num_samples = []
        self.cum_samples = []
        total = 0
        for path in lmdb_paths:
            env = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
            with env.begin(write=False) as txn:
                n = int(txn.get('num-samples'.encode()))
            self.envs.append(env)
            self.num_samples.append(n)
            total += n
            self.cum_samples.append(total)
        self.total_samples = total

        self.size = size
        self.transform = T.Compose([
            T.Resize((self.size[0], self.size[1])),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.total_samples

    def _get_env_and_idx(self, index):
        for env_idx, cum in enumerate(self.cum_samples):
            if index < cum:
                if env_idx == 0:
                    local_idx = index
                else:
                    local_idx = index - self.cum_samples[env_idx - 1]
                return self.envs[env_idx], local_idx + 1
        raise IndexError

    def __getitem__(self, index):
        env, idx = self._get_env_and_idx(index)
        with env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % idx
            orilabel_key = 'label-%09d'.encode() % idx
            editlabel_key = 'editlabel-%09d'.encode() % idx
            imgbuf = txn.get(img_key)
            editlabel = txn.get(editlabel_key).decode('utf-8')
            ori_label = txn.get(orilabel_key).decode('utf-8')

        img = Image.open(six.BytesIO(imgbuf)).convert('RGB')
        w, h = img.size
        if h > 2 * w:
            img = img.rotate(90, expand=True)
        img_tr = self.transform(img)
        return dict(img=img_tr, text=editlabel, ori_label=ori_label, pil_img=imgbuf)


class InferPipeline:
    def __init__(self, model):
        self.model = model
        self.scheduler = model.noise_scheduler
        self.device = model.device
        self.unet = model.unet
        self.vae = model.vae
        self.control_model = model.control_model
        self.text_encoder = model.text_encoder
        self.NORMALIZER = model.NORMALIZER

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float = 0.0,
        verbose=False,
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.NORMALIZER * latents.detach()
        image = self.model.vae.decode(latents).sample
        if return_type == 'np':
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == 'pt':
            image = image.clamp(0, 1)
        return image

    @torch.no_grad()
    def __call__(
            self,
            hint,
            cond,
            height=64,
            width=256,
            num_inference_steps=50,
            guidance_scale=2,
            latents=None,
            **kwds):

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        batch_size = len(cond)
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint

            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)
            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input,
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.step(noise_pred, t, latents)

        image = self.latent2image(latents, return_type="pt")
        return image  # [B,C,H,W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="textriro.pth")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--input_lmdb", type=str, default="RIROCommon/Equal/IC13")
    parser.add_argument("--output_lmdb", type=str, default="RIROOut/Equal/IC13")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--image_width", type=int, default=256)
    args = parser.parse_args()

    seed_everything(args.seed)

    model = create_model(args.config).cuda()
    model.load_state_dict(load_state_dict(args.ckpt_path), strict=False)
    model.eval()
    pipeline = InferPipeline(model)

    dataset = LmdbTextDataset(args.input_lmdb, size=[args.image_height, args.image_width])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    writer = LmdbWriter(args.output_lmdb)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            hint = batch['img'].cuda()
            cond = batch['text']
            ori_labels = batch['ori_label']
            pil_imgs = batch['pil_img']
            gen_imgs = pipeline(
                hint,
                cond,
                height=args.image_height,
                width=args.image_width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            gen_imgs = gen_imgs.clamp(0, 1)
            for i in range(len(cond)):
                orig_pil = Image.open(six.BytesIO(pil_imgs[i])).convert('RGB')
                writer.write_sample(orig_pil, cond[i], gen_imgs[i], ori_labels[i])
    writer.close()


if __name__ == "__main__":
    main()
