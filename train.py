import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import wandb
import random
import logging
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import tifffile
from skimage.transform import resize

from model import Rec_Transformer
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader

writer = SummaryWriter('log')
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(cfg, model):
    # torch.save(model.module.state_dict(), cfg.dir.save_model_dir)
    torch.save(model.state_dict(), os.path.join(cfg.dir.save_model_dir, "best_model.pth"))


def load_model(cfg, model):
    loaded_dict = torch.load(cfg.dir.load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)


def setup(cfg):
    model = Rec_Transformer(input_size=(cfg.basic.H, cfg.basic.W), rec_size=(cfg.basic.H, cfg.basic.W))               # TODO: added args
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    if cfg.train.load == True:
        load_model(cfg, model)

    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(cfg):
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed_all(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)
    torch.backends.cudnn.deterministic = True


def valid(cfg, model, val_loader, global_step, val_psnr, val_ssim):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    MSE = torch.nn.MSELoss()
    MSE.cuda()

    total_mse = 0.0
    total_loss = 0.0

    # Ensure metrics are reset
    val_psnr.reset()
    val_ssim.reset()

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)
        x, y = batch
        with torch.no_grad():
            outputs = model(x)

            outputs, y = crop_borders((outputs, y), cfg.basic.dataset, cfg.basic.downsize_coeff, batch=True)

            MSE_loss=MSE(outputs, y.to(torch.float))
            eval_loss = cfg.loss.MSE_t * MSE_loss
            eval_losses.update(eval_loss.item())
            total_loss += eval_loss.item()
            total_mse += MSE_loss.item()

            output_detached = outputs.detach()
            target_detached = y.detach()

            val_psnr.update(output_detached, target_detached)
            val_ssim.update(output_detached, target_detached)

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # Compute Averages
    avg_psnr = val_psnr.compute().item()
    avg_ssim = val_ssim.compute().item()
    avg_mse = total_mse / len(val_loader.dataset) # Or len(val_loader) if batch_avg, keeping your style
    avg_loss = total_loss / len(val_loader)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    return eval_losses.avg, avg_psnr, avg_mse, avg_ssim, avg_loss


def train(cfg, debug):
    """ Train the model """
    model = setup(cfg)
    #freezing layers
    '''
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.Decoder.parameters():
        parameter.requires_grad = True
    '''
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    untrainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
    print(untrainable_params)

    model.cuda()
    #model = torch.nn.DataParallel(model)

    # Prepare dataset
    train_loader, val_loader = get_loader(cfg)    

    # Prepare optimizer and scheduler
    if cfg.optimizer.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.optimizer.learning_rate,
                                    momentum=0.9,
                                    weight_decay=cfg.optimizer.weight_decay)
    if cfg.optimizer.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.optimizer.learning_rate,
                                      weight_decay=cfg.optimizer.weight_decay)
    t_total = cfg.train.num_steps

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.scheduler.warmup_steps, t_total=t_total)

    model.zero_grad()

    train_psnr = PSNR(data_range=1.0).cuda()
    train_ssim = SSIM(data_range=1.0).cuda()

    val_psnr = PSNR(data_range=1.0).cuda()
    val_ssim = SSIM(data_range=1.0).cuda()

    # Placeholders for WandB to avoid crashing if validation hasn't run yet
    val_psnr_out, val_mse_loss, val_ssim_out, val_loss_out = 0, 0, 0, 0

    losses = AverageMeter()
    MSE = torch.nn.MSELoss()
    MSE.cuda()
    global_step=0
    best_losses = 999999

    interval_loss = 0.0
    interval_mse = 0.0
    interval_steps = 0

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", cfg.train.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.train.train_batch_size)
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)
            x, y = batch

            outputs = model(x)
            outputs, y = crop_borders((outputs, y), cfg.basic.dataset, cfg.basic.downsize_coeff, batch=True)

            MSE_loss = MSE(outputs, y.to(torch.float))
            loss = cfg.loss.MSE_t * MSE_loss
            loss.mean().backward()
            losses.update(loss.mean().item())

            with torch.no_grad():
                output_detached = outputs.detach()
                target_detached = y.detach()

                train_psnr.update(output_detached, target_detached)
                train_ssim.update(output_detached, target_detached)
                
                # Manual accumulation for things that aren't torchmetrics objects
                interval_loss += loss.mean().item()
                interval_mse += MSE_loss.item()
                
                interval_steps += 1

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if cfg.scheduler.use==True:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

            if global_step % 1000 == 0:         # TODO: added this if - VP
                torch.save(model.state_dict(), os.path.join(cfg.dir.save_model_dir, 'latest_model.pth'))

                train_psnr_out = train_psnr.compute().item()
                train_ssim_out = train_ssim.compute().item()
                train_loss_out = interval_loss / interval_steps
                train_mse_loss = interval_mse / interval_steps

                if not debug:
                    wandb_data = {"training_mse_loss":train_mse_loss, 
                            "training_psnr": train_psnr_out,
                            "train_ssim": train_ssim_out,
                            "train_mse": train_mse_loss,
                            "val_mse": val_mse_loss,
                            "val_psnr": val_psnr_out, 
                            "val_ssim": val_ssim_out,
                            "val_loss": val_loss_out,
                            "epoch":global_step, 
                            "learning rate":optimizer.param_groups[-1]['lr']}
                    if val_psnr_out != 0:
                        wandb_data.update({
                            "val_mse": val_mse_loss,
                            "val_psnr": val_psnr_out, 
                            "val_ssim": val_ssim_out,
                            "val_loss": val_loss_out,
                        })
                    wandb.log(wandb_data)

                # Reset metrics for next interval
                train_psnr.reset()
                train_ssim.reset()
                interval_loss = 0.0
                interval_mse = 0.0
                interval_steps = 0

            if global_step % cfg.train.eval_every == 0:
                #val_loss_out, val_psnr_out, val_mse_loss, val_ssim_out, val_loss_out_duplicate=valid(cfg, model, val_loader, global_step, val_psnr, val_ssim)

                if val_mse_loss < best_losses:
                    torch.save(model.state_dict(), os.path.join(cfg.dir.save_model_dir, "best_model.pth"))
                    best_losses = val_mse_loss
                    print(f"New best model saved at step {global_step}")

                # save a validation rec image. 
                # save an image from validation run as visual reference while running model - VP
                ground_truth = tifffile.imread(cfg.dir.val_pattern_gt_dir)
                test_in = tifffile.imread(cfg.dir.val_pattern_dir)
                test_in = (test_in/255).astype(np.float32)     # max of measurements is 255. Normalizing to range [0, 1]
                height, width, _ = test_in.shape
                test_in = resize(test_in, (height // cfg.basic.downsize_coeff, width // cfg.basic.downsize_coeff), anti_aliasing=True).astype(np.float32) 
                test_in = np.clip(test_in, 0,1)
                
                test_out = np.zeros((cfg.basic.H, cfg.basic.W, 3))
                for c in range(3):
                    test_in_one_channel = test_in[:, :, c]
                    test_in_one_channel = np.reshape(test_in_one_channel,
                                                     (1, 1, cfg.basic.H, cfg.basic.W))
                    test_in_one_channel = (test_in_one_channel - test_in_one_channel.mean()) / test_in_one_channel.std()
                    test_in_one_channel.astype(float)
                    test_in_one_channel = torch.from_numpy(test_in_one_channel)
                    test_in_one_channel.cuda()
                    test_out_one_channel = model(test_in_one_channel)
                    test_out_one_channel, _ = crop_borders((test_out_one_channel, test_out_one_channel), cfg.basic.dataset, cfg.basic.downsize_coeff, batch=True)
                    test_out_one_channel = test_out_one_channel[0][0].to('cpu').detach().numpy().copy()
                    test_out_one_channel = np.clip(test_out_one_channel, 0, 1)

                    if c == 0:
                        h_crop, w_crop = test_out_one_channel.shape
                        test_out = np.zeros((h_crop, w_crop, 3))
                    test_out[:, :, c] = test_out_one_channel
                # cv2.imwrite(cfg.dir.val_rec_dir + str(global_step) + '.bmp',
                #             cv2.normalize(test_out, None, 0, 255, cv2.NORM_MINMAX))

                if not debug:
                    wandb.log({
                        "reconstructed butterfly": wandb.Image(test_out),
                        "ground truth butterfly": wandb.Image(ground_truth),
                        "val_step": global_step # distinct step for Image slider
                    })
                model.train()

            if global_step % t_total == 0:
                break
        losses.reset()

        if global_step % t_total == 0:
            break

    logger.info("Best Loss: \t%f" % best_losses)
    logger.info("End Training!")

def crop_borders(img, dataset, downsize_coeff, batch=False):

    if batch:
        output, target = img
        assert output.ndim == 4 and target.ndim == 4, \
            f"Expected batched tensors of shape (B,C,H,W), got {output.shape} and {target.shape}"

        if dataset == "mirflickr":
            output = output[:,:,60:,62:-38]
            target = target[:,:,60:,62:-38]
        elif dataset == "diffusercam":    
            if downsize_coeff == 8:             
                output = output[:,:,:134, 56:191]     
                target = target[:,:,:134, 56:191]  
            elif downsize_coeff == 4:
                output = output[:,:,:267, 111:379]
                target = target[:,:,:267, 111:379]
        elif dataset == "rml":
            if downsize_coeff == 8:
                output = output[:,:,14:134, 61:181]
                target = target[:,:,14:134, 61:181]
            elif downsize_coeff == 4:
                output = output[:,:,27:267, 121:361]
                target = target[:,:,27:267, 121:361]
        return output, target

    assert img.ndim == 3, f"Expected (H,W,C) or (C,H,W), got {img.shape}"

    if dataset == "mirflickr":
        img = img[60:,62:-38,:]
    elif dataset == "diffusercam":
        if downsize_coeff == 8:
            img = img[:134, 56:191,:]  
        elif downsize_coeff == 4:
            img = img[:267, 111:379,:]
    elif dataset == "rml":
        if downsize_coeff == 8:
            img = img[14:134, 61:181,:]
        elif downsize_coeff == 4:
            img = img[27:267, 121:361,:]
    return img

def main():       
    #dist.init_process_group(backend='nccl')  
    cfg = OmegaConf.load('/home/ponoma/workspace/Pan_Transformer/configs.yaml')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(cfg)
    if not os.path.exists(cfg.dir.save_model_dir):          # TODO: added this for saving purposes - VP
        os.mkdir(cfg.dir.save_model_dir)

    debug = False

    wandb_id = None
    run_name = "pan_0.5_rml_6_x4_downsize"
    if not debug:
        run = wandb.init(project="convnext", 
                         name=run_name, 
                         id=wandb_id,            # If this is None, W&B creates a new ID. If it's a string, it resumes that ID.
                         resume="allow",         # "allow" means: if ID exists, resume it; if not, create new.
                         config={"architecture": "Pan Transformer"})  

    '''
    model = setup(cfg)
    print(model)
    model.cuda()
    dummy_input = torch.rand(1, 1, 1600, 1600).cuda()
    with SummaryWriter(comment='Rec_Transformer') as w:
        w.add_graph(model, (dummy_input.to(torch.float),))
    '''
    train(cfg, debug)


if __name__ == "__main__":
    main()
