import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
import random
from PIL import Image
import io
import cv2
from scipy.ndimage import gaussian_filter
from noiseprobe.noiseprobe.registry import registry
from noiseprobe.noiseprobe.base import BaseProbe

# 1. Noise perturbations
class GaussianNoiseProbeImage(BaseProbe):
    name = 'gaussian_noise_image'

    def __call__(self, X: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(X) * std
        return X + noise
    
    def get_name(self):
        return 'gaussian_noise_image'

class SaltAndPepperNoiseProbeImage(BaseProbe):
    name = 'salt_and_pepper_noise_image'

    def __call__(self, X: torch.Tensor, prob: float = 0.05) -> torch.Tensor:
        out = X.clone()
        mask = torch.rand_like(X) < prob
        salt = (torch.rand_like(X) < 0.5) & mask
        pepper = (~salt) & mask
        out[salt] = 1.0
        out[pepper] = 0.0
        return out
    
    def get_name(self):
        return 'salt_and_pepper_noise_image'

class SpeckleNoiseProbeImage(BaseProbe):
    name = 'speckle_image_noise'

    def __call__(self, X: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(X) * std
        return X + X * noise
    
    def get_name(self):
        return 'speckle_image_noise'

# 2. Blurring
class GaussianBlurProbeImage(BaseProbe):
    name = 'gaussian_blur_image'

    def __call__(self, X: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))
        return blur(X)
    
    def get_name(self):
        return 'gaussian_blur_image'

class MotionBlurProbeImage(BaseProbe):
    name = 'motion_blur_image'

    def __call__(self, X: torch.Tensor, kernel_size: int = 15, angle: float = 0.0) -> torch.Tensor:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel /= kernel_size
        np_img = X.permute(1, 2, 0).cpu().numpy()
        blurred = cv2.filter2D(np_img, -1, kernel)
        return torch.from_numpy(blurred).permute(2, 0, 1).to(X.device)
    
    def get_name(self):
        return 'motion_blur_image'

class MedianBlurProbeImage(BaseProbe):
    name = 'median_blur_image'

    def __call__(self, X: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        np_img = (X.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        med = cv2.medianBlur(np_img, kernel_size)
        return torch.from_numpy(med.astype(np.float32) / 255).permute(2, 0, 1).to(X.device)
    
    def get_name(self):
        return 'median_blur_image'

# 3. Geometric transforms
class RotateProbeImage(BaseProbe):
    name = 'rotate_image'

    def __call__(self, X: torch.Tensor, angle: float) -> torch.Tensor:
        return TF.rotate(X, angle)
    
    def get_name(self):
        return 'rotate_image'

class TranslateProbeImage(BaseProbe):
    name = 'translate_image'

    def __call__(self, X: torch.Tensor, translate: tuple) -> torch.Tensor:
        return TF.affine(X, angle=0, translate=translate, scale=1, shear=0)

    def get_name(self):
        return 'translate_image'

class ScaleProbeImage(BaseProbe):
    name = 'scale_image'

    def __call__(self, X: torch.Tensor, scale: float) -> torch.Tensor:
        h, w = X.shape[1], X.shape[2]
        return TF.resize(X, [int(h*scale), int(w*scale)])

    def get_name(self):
        return 'scale_image'

class ShearProbeImage(BaseProbe):
    name = 'shear_image'

    def __call__(self, X: torch.Tensor, shear: tuple) -> torch.Tensor:
        return TF.affine(X, angle=0, translate=(0,0), scale=1, shear=shear)

    def get_name(self):
        return 'shear_image'

class ElasticDeformationProbeImage(BaseProbe):
    name = 'elastic_deformation_image'

    def __call__(self, X: torch.Tensor, alpha: float = 34, sigma: float = 4) -> torch.Tensor:
        np_img = X.permute(1, 2, 0).cpu().numpy()
        shape = np_img.shape[:2]
        rs = np.random.RandomState(None)
        dx = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma) * alpha
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)
        warped = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return torch.from_numpy(warped).permute(2, 0, 1).to(X.device)

    def get_name(self):
        return 'elastic_deformation_image'

# 4. Color & intensity
class BrightnessProbeImage(BaseProbe):
    name = 'brightness_image'

    def __call__(self, X: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_brightness(X, factor)

    def get_name(self):
        return 'brightness_image'

class ContrastProbeImage(BaseProbe):
    name = 'contrast_image'

    def __call__(self, X: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_contrast(X, factor)
    
    def get_name(self):
        return 'contrast_image'

class SaturationProbeImage(BaseProbe):
    name = 'saturation_image'

    def __call__(self, X: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_saturation(X, factor)
    
    def get_name(self):
        return 'saturation_image'

class HueProbeImage(BaseProbe):
    name = 'hue_image'

    def __call__(self, X: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_hue(X, factor)

    def get_name(self):
        return 'hue_image'

class GammaCorrectionProbeImage(BaseProbe):
    name = 'gamma_image'

    def __call__(self, X: torch.Tensor, gamma: float) -> torch.Tensor:
        return TF.adjust_gamma(X, gamma)

    def get_name(self):
        return 'gamma_image'

# 5. Occlusion & dropout
class RandomErasingProbeImage(BaseProbe):
    name = 'random_erasing_image'

    def __call__(self, X: torch.Tensor, p: float = 0.5, scale: tuple = (0.02,0.33), ratio: tuple = (0.3,3.3)) -> torch.Tensor:
        eraser = T.RandomErasing(p=p, scale=scale, ratio=ratio, value=0)
        return eraser(X)

    def get_name(self):
        return 'random_erasing_image'

class PatchOcclusionProbeImage(BaseProbe):
    name = 'patch_occlusion_image'

    def __call__(self, X: torch.Tensor, size: tuple = (50,50)) -> torch.Tensor:
        out = X.clone()
        _, h, w = out.shape
        th, tw = size
        top = random.randint(0, h-th)
        left = random.randint(0, w-tw)
        out[:, top:top+th, left:left+tw] = 0
        return out

    def get_name(self):
        return 'patch_occlusion_image'

class CoarseDropoutProbeImage(BaseProbe):
    name = 'coarse_dropout_image'

    def __call__(self, X: torch.Tensor, holes: int = 5, size: tuple = (20,20)) -> torch.Tensor:
        out = X.clone()
        _, h, w = out.shape
        th, tw = size
        for _ in range(holes):
            top = random.randint(0, h-th)
            left = random.randint(0, w-tw)
            out[:, top:top+th, left:left+tw] = 0
        return out
    
    def get_name(self):
        return 'coarse_dropout_image'

# 6. Compression artifacts
class JpegCompressionProbeImage(BaseProbe):
    name = 'jpeg_compression_image'

    def __call__(self, X: torch.Tensor, quality: int = 50) -> torch.Tensor:
        pil = TF.to_pil_image(X)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        comp = Image.open(buf)
        return TF.to_tensor(comp).to(X.device)

class DownsampleUpsampleProbeImage(BaseProbe):
    name = 'down_upsample_image'

    def __call__(self, X: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        h, w = X.shape[1], X.shape[2]
        small = TF.resize(X, [int(h*scale), int(w*scale)])
        return TF.resize(small, [h, w])
    
    def get_name(self):
        return 'down_upsample_image'

# 7. Pixel-level distortions
class ColorQuantizationProbeImage(BaseProbe):
    name = 'color_quantization_image'

    def __call__(self, X: torch.Tensor, bits: int = 4) -> torch.Tensor:
        levels = 2 ** bits
        return torch.floor(X * levels) / (levels - 1)
    
    def get_name(self):
        return 'color_quantization_image'

class PixelationProbeImage(BaseProbe):
    name = 'pixelation_image'

    def __call__(self, X: torch.Tensor, block_size: int = 16) -> torch.Tensor:
        _, h, w = X.shape
        small = TF.resize(X, [h//block_size, w//block_size])
        return TF.resize(small, [h, w])
    
    def get_name(self):
        return 'pixelation_image'

# 8. Adversarial attacks
class FGSMAttackProbeImage(BaseProbe):
    name = 'fgsm_attack_image'

    def __call__(self, X: torch.Tensor, label: torch.Tensor, loss_fn: callable, epsilon: float = 0.01) -> torch.Tensor:
        adv = X.clone().detach().requires_grad_(True)
        out = self.model(adv.unsqueeze(0))
        loss = loss_fn(out, label.unsqueeze(0))
        loss.backward()
        return (adv + epsilon * adv.grad.sign()).detach()

    def get_name(self):
        return 'fgsm_attack_image'

class PGDAttackProbeImage(BaseProbe):
    name = 'pgd_attack_image'

    def __call__(self, X: torch.Tensor, label: torch.Tensor, loss_fn: callable,
                 epsilon: float = 0.03, alpha: float = 0.01, iters: int = 10) -> torch.Tensor:
        adv = X.clone().detach()
        orig = X.clone().detach()
        for _ in range(iters):
            adv.requires_grad_(True)
            out = self.model(adv.unsqueeze(0))
            loss = loss_fn(out, label.unsqueeze(0))
            loss.backward()
            tmp = adv + alpha * adv.grad.sign()
            eta = torch.clamp(tmp - orig, -epsilon, epsilon)
            adv = torch.clamp(orig + eta, 0, 1).detach()
        return adv

    def get_name(self):
        return 'pgd_attack_image'
