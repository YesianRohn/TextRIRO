import torch
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from utils import create_model, load_state_dict
from pytorch_lightning import seed_everything
import numpy as np
from PIL import Image
import os

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

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            assert image.dim() == 4, print("input dims should be 4 !")
            latents = self.vae.encode(image.to(self.device)).latent_dist.sample()
            latents = latents * self.NORMALIZER
        return latents

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

    def latent2image_grad(self, latents):
        latents = 1 / self.NORMALIZER * latents
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def __call__(
            self,
            hint,
            cond,
            batch_size=1,
            height=64,
            width=256,
            num_inference_steps=50,
            guidance_scale=2,
            eta=0.0,
            latents=None,
            unconditioning=None,
            ref_intermediate_latents=None,
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
        else:
            pass

        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(self.scheduler.timesteps):
            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint
                
            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)

            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
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


def load_image(image_path, image_height=64, image_width=256):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open(image_path)
    image = T.ToTensor()(T.Resize((image_height, image_width))(img.convert("RGB")))
    image = image.to(device)
    return image.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="textriro.pth")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=2.2)
    parser.add_argument("--dataset_dir", type=str, default="example")
    parser.add_argument("--output_dir", type=str, default="example/result/")
    args = parser.parse_args()

    cfg_path = 'configs/train.yaml'
    model = create_model(cfg_path).cuda()
    model.load_state_dict(load_state_dict(args.ckpt_path), strict=False)
    model.eval()
    pipeline = InferPipeline(model)

    dataset_dir = args.dataset_dir
    style_dir = os.path.join(dataset_dir, 'i_s')
    style_images_path = {image_name: os.path.join(style_dir, image_name) for image_name in os.listdir(style_dir)}

    target_txt = os.path.join(dataset_dir, 'i_t.txt')
    target_dict = {}
    with open(target_txt, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                image_name, text = line.strip().split(' ', 1)
                target_dict[image_name] = text

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    seed = args.seed
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    seed_everything(seed)
    for i in tqdm(range(len(list(style_images_path.keys())))):
        image_name = list(style_images_path.keys())[i]
        image_path = style_images_path[image_name]
        target_text = target_dict[image_name]
        w,h = Image.open(image_path).size
        style_image = load_image(image_path)
        result = pipeline(hint=style_image, cond=[target_text],
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
        )[0]
        result = result.clamp(0, 1).cpu().numpy()
        result = np.transpose(result, (1,2,0))
        result = Image.fromarray((result * 255).astype(np.uint8)).resize((w, h))
        result.save(os.path.join(output_dir, image_name))

if __name__ == "__main__":
    main()
