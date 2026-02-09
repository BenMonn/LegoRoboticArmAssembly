#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import mujoco

from PIL import Image, ImageEnhance, ImageFilter
from stack_env import StackEnv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_reset_obs(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def render_rgb(model, data, camera_name: str, width: int, height: int):
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()  # uint8 HxWx3
    renderer.close()
    return rgb

def get_body_xy_z(model, data, body_name: str):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[bid].copy()   # world position (x, y, z) of the body frame
    return float(pos[0]), float(pos[1]), float(pos[2])


# SimCLR-style augmentations
def try_build_torchvision_aug(img_size: int):
    try:
        import torchvision.transforms as T

        # Classic SimCLR-ish augmentation recipe (reasonable defaults)
        aug = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # Gaussian blur is common in SimCLR; kernel size should be odd
            T.RandomApply([T.GaussianBlur(kernel_size=max(3, (img_size // 10) * 2 + 1), sigma=(0.1, 2.0))], p=0.5),
        ])
        return aug
    except Exception:
        return None


def pil_fallback_aug(pil_img: Image.Image, out_size: int, rng: np.random.Generator):
    # Random crop (center-ish) + resize
    w, h = pil_img.size
    scale = float(rng.uniform(0.65, 1.0))
    cw, ch = int(w * scale), int(h * scale)
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    pil_img = pil_img.crop((x0, y0, x0 + cw, y0 + ch)).resize((out_size, out_size), Image.BILINEAR)

    # Color jitter-ish
    if rng.random() < 0.8:
        pil_img = ImageEnhance.Brightness(pil_img).enhance(float(rng.uniform(0.7, 1.3)))
        pil_img = ImageEnhance.Contrast(pil_img).enhance(float(rng.uniform(0.7, 1.3)))
        pil_img = ImageEnhance.Color(pil_img).enhance(float(rng.uniform(0.7, 1.3)))

    # Grayscale sometimes
    if rng.random() < 0.2:
        pil_img = pil_img.convert("L").convert("RGB")

    # Blur sometimes
    if rng.random() < 0.5:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 1.2))))

    # Horizontal flip
    if rng.random() < 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    return pil_img

def get_body_xy_z(model, data, body_name: str):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[bid].copy()
    return float(pos[0]), float(pos[1]), float(pos[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML (scene/model).")
    parser.add_argument("--out", type=str, default="simclr_pairs", help="Output directory.")
    parser.add_argument("--camera", type=str, default="global_cam", help="MuJoCo camera name.")
    parser.add_argument("--pairs", type=int, default=5000, help="Number of positive pairs to save.")
    parser.add_argument("--render_w", type=int, default=256, help="Render width.")
    parser.add_argument("--render_h", type=int, default=256, help="Render height.")
    parser.add_argument("--img_size", type=int, default=224, help="Final saved image size (square).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    ensure_dir(args.out)
    img_dir = os.path.join(args.out, "images")
    ensure_dir(img_dir)

    meta_path = os.path.join(args.out, "pairs.csv")

    rng = np.random.default_rng(args.seed)

    # Build env
    env = StackEnv(args.xml)

    # Build augmentation pipeline
    tv_aug = try_build_torchvision_aug(args.img_size)
    use_torchvision = tv_aug is not None
    print(f"[collect_pairs] torchvision augmentations: {'ON' if use_torchvision else 'OFF (PIL fallback)'}")

    # Open metadata CSV
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "img_a", "img_b", "reset_mode", "final_label", "x", "y", "yaw_deg"])

        for i in range(args.pairs):
            # Reset env
            _ = get_reset_obs(env)

            # Render one base frame
            rgb = render_rgb(env.model, env.data, args.camera, args.render_w, args.render_h)
            base = Image.fromarray(rgb)

            # Produce 2 augmented views (positive pair)
            if use_torchvision:
                a = tv_aug(base)
                b = tv_aug(base)
            else:
                a = pil_fallback_aug(base, args.img_size, rng)
                b = pil_fallback_aug(base, args.img_size, rng)

            # Always ensure final size exactly img_size
            a = a.resize((args.img_size, args.img_size), Image.BILINEAR)
            b = b.resize((args.img_size, args.img_size), Image.BILINEAR)

            # Save
            img_a_name = f"pair_{i:07d}_a.png"
            img_b_name = f"pair_{i:07d}_b.png"
            a_path = os.path.join(img_dir, img_a_name)
            b_path = os.path.join(img_dir, img_b_name)
            a.save(a_path)
            b.save(b_path)

            # Optional: log rough stack pose (based on brick1)
            # log xy position and yaw angle of brick1
            b1 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "brick1")
            x, y = float(env.data.xpos[b1][0]), float(env.data.xpos[b1][1])

            # data.xquat is wxyz; yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) for ZYX convention
            qw, qx, qy, qz = [float(v) for v in env.data.xquat[b1]]
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            yaw_deg = float(np.rad2deg(yaw))

            x1, y1, z1 = get_body_xy_z(env.model, env.data, "brick1")
            x2, y2, z2 = get_body_xy_z(env.model, env.data, "brick2")

            BRICK_H = 0.023
            z_tol = 0.004 
            xy_tol = 0.010

            dz = z2 - z1
            dxy = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            final_label = "stacked" if (abs(dz - BRICK_H) < z_tol and dxy < xy_tol) else "not_stacked"
            reset_mode = getattr(env, "last_reset_mode", "unknown")

            mode = getattr(env, "last_reset_mode", "unknown")
            writer.writerow([i, os.path.join("images", img_a_name), os.path.join("images", img_b_name), reset_mode, final_label, x, y, yaw_deg])

            if (i + 1) % 100 == 0:
                print(f"[collect_pairs] saved {i+1}/{args.pairs} pairs -> {args.out}")

    print(f"[collect_pairs] Done. Images in: {img_dir}")
    print(f"[collect_pairs] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
