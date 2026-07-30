from model import Rec_Transformer
import torch
import numpy as np
import tifffile
import os
import argparse
import kornia.geometry.transform as transform
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import pickle
import re

from model import Rec_Transformer
from utils.data_utils import get_test_loader

rml_homography_matrix_path = "" # put in path to rml homography matrix
diffuser_homography_matrix_path = "" # put in path to diffuser homography matrix


def _extract_image_id(name: str):
    """Match ids in filenames like ...img_64... or im64."""
    patterns = [
        r"(?:^|_)img_(\d+)(?:_|$)",
        r"(?:^|_)im(\d+)(?:_|$)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    m = re.search(r"\d+", name)
    return int(m.group(0)) if m else None


def confidence_interval_list(data_list, confidence_interval=0.95):
    error_lo = np.percentile(data_list, 100 * (1 - confidence_interval) / 2)
    error_hi = np.percentile(data_list, 100 * (1 - (1 - confidence_interval) / 2))
    mean = np.mean(data_list)
    return error_lo, error_hi, mean


def _run_name(cfg):
    if cfg.basic.dataset != "mirflickr":
        return (
            f"pan_{cfg.basic.dataset_size}_{cfg.basic.dataset}_{cfg.train.train_batch_size}_"
            f"x{cfg.basic.downsize_coeff}_downsize_{cfg.optimizer.learning_rate}_lr"
        )
    return f"pan_mirflickr_{cfg.train.train_batch_size}_big_gpu"


def parse_args():
    p = argparse.ArgumentParser(description="Pan_Transformer inference")
    p.add_argument("--config", type=str, default="configs.yaml", help="Path to OmegaConf YAML")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override infer.batch_size for the test DataLoader only.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.infer.gpu_visible_id)
    gpu_num = int(cfg.infer.gpu_num)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_num)
    else:
        device = torch.device("cpu")

    run_name = _run_name(cfg)
    save_dir = os.path.join(cfg.infer.save_infer_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_id_list_raw = getattr(cfg.infer, "save_ids", None) or []
    save_id_list = [int(x) for x in save_id_list_raw]

    input_size = (cfg.basic.H, cfg.basic.W)
    model = Rec_Transformer(input_size=input_size, rec_size=input_size)
    model.to(device)

    model_path = os.path.join(cfg.dir.load_model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    loaded_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    global_step = checkpoint.get("global_step", "unknown")
    print(f"Loaded (partial) weights from {model_path} (step {global_step})")

    # Test loader: optional CLI batch-size override
    if args.batch_size is not None:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"infer": {"batch_size": int(args.batch_size)}})
        )
    test_loader = get_test_loader(cfg)

    dataset = cfg.basic.dataset
    if dataset != 'mirflickr':
        if dataset == "rml":            
            homography_matrix = torch.load(rml_homography_matrix_path, weights_only=True) 
        elif dataset == 'diffuser':
            homography_matrix = torch.load(diffuser_homography_matrix_path, weights_only=True)

    # Need to invert to get RML/diffuser -> GT warp
    imager_to_gt_homography_matrix = torch.inverse(homography_matrix).to(device)

    psnr = PSNR(data_range=1.0).to(device)
    ssim = SSIM(data_range=1.0).to(device)
    lpips = LPIPS(net_type="alex", normalize=True).to(device)

    mse_list, psnr_list, ssim_list, lpips_list = [], [], [], []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc="test"):
            if len(batch) == 3:
                input, target, img_names = batch
            else:
                input, target = batch
                img_names = None

            input = input.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)

            # input/target: (B, 3, H, W) — run single-channel model per color channel
            outs = []
            for c in range(3):
                outs.append(model(input[:, c : c + 1, :, :]))
            output = torch.cat(outs, dim=1)
            output = torch.clamp(output, 0.0, 1.0)

            output = transform.warp_perspective(
                output.float(),
                imager_to_gt_homography_matrix,
                dsize=(output.shape[2], output.shape[3]),
            )
            output = torch.clamp(output, 0.0, 1.0)
            target = transform.warp_perspective(
                target.float(),
                imager_to_gt_homography_matrix,
                dsize=(target.shape[2], target.shape[3]),
            )
            target = torch.clamp(target, 0.0, 1.0)

            output = output[:,:, 52:266, 129:343]          
            target = target[:,:, 52:266, 129:343]

            # Save selected ids (per batch element)
            name = img_names[0]
            img_id = _extract_image_id(name)
            should_save = img_id is not None and img_id in save_id_list
            if should_save:
                out_img = output.squeeze().cpu().detach().numpy()
                out_img = np.moveaxis(out_img, 0, -1)
                np.save(os.path.join(save_dir, str(img_id) + ".npy"), out_img)

                out_target = target.squeeze().cpu().detach().numpy()
                out_target = np.moveaxis(out_target, 0, -1)
                np.save(os.path.join(save_dir, str(img_id) + "_gt.npy"), out_target)

            psnr_val = psnr(output, target)
            ssim_val = ssim(output, target)
            mse_val = torch.nn.functional.mse_loss(output, target, reduction="mean")
            clipped_out = torch.clamp(output, min=0.0, max=1.0)
            lpips_val = lpips(clipped_out, target)

            psnr.reset()
            ssim.reset()
            lpips.reset()

            psnr_list.append(psnr_val.item())
            ssim_list.append(ssim_val.item())
            mse_list.append(mse_val.item())
            lpips_list.append(lpips_val.item())

    n = len(mse_list)
    if n == 0:
        raise RuntimeError("No batches in test_loader — check test_patterns.npy / test_targets.npy.")

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_mse = float(np.mean(mse_list))
    mean_lpips = float(np.mean(lpips_list))

    final_results = {
        "avg_mse": mean_mse,
        "avg_lpips": mean_lpips,
        "avg_psnr": mean_psnr,
        "avg_ssim": mean_ssim,
        "mse_per_batch": mse_list,
        "lpips_per_batch": lpips_list,
        "psnr_per_batch": psnr_list,
        "ssim_per_batch": ssim_list,
        "confidence_interval_mse": confidence_interval_list(mse_list),
        "confidence_interval_lpips": confidence_interval_list(lpips_list),
        "confidence_interval_psnr": confidence_interval_list(psnr_list),
        "confidence_interval_ssim": confidence_interval_list(ssim_list),
    }

    if cfg.infer.save_metrics:
        np.save(f"{save_dir}/metrics_list.npy", final_results)

    print(f"\nCheckpoint: {model_path}")
    print(f"Avg. MSE: {mean_mse}")
    print(f"Avg. PSNR: {mean_psnr}")
    print(f"Avg. SSIM: {mean_ssim}")
    print(f"Avg. LPIPS: {mean_lpips}")
    print(f"Saved outputs under: {save_dir}")


if __name__ == "__main__":
    main()
