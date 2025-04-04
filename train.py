import torch
from torch.utils.data import DataLoader
from torch import nn
import argparse
from tqdm import tqdm
from torch.nn import functional as F
from model import SimpleConv, SANet, TransformerNet
from dataset import SaliencyDataset, process_batch, process_batch_2
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 18
MEDIUM_SIZE = 22
LARGE_SIZE = 26
plt.rcParams.update({
    'font.size': MEDIUM_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': LARGE_SIZE,
    'xtick.labelsize': SMALL_SIZE ,
    'ytick.labelsize': SMALL_SIZE ,
    'legend.fontsize': SMALL_SIZE-2,
    'figure.titlesize': LARGE_SIZE,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'grid.linewidth': 1,
    'grid.linestyle': ':',
    'lines.linewidth': 2,
    'lines.markersize': 6
})


def norm_minmax(x, dim, min_val=None, max_val=None):
    """
    Applies Min-Max Normalization along a specified dimension.
    
    Args:
        x (Tensor): Input tensor.
        dim (int): Dimension along which to compute min/max.
        min_val (Tensor or None): Minimum value (if None, computed from x).
        max_val (Tensor or None): Maximum value (if None, computed from x).
    
    Returns:
        Tensor: Normalized tensor.
    """
    if min_val is None:
        min_val, _ = x.min(dim=dim, keepdim=True)
    if max_val is None:
        max_val, _ = x.max(dim=dim, keepdim=True)
    
    return (x - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    

def plot_and_save_losses(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot.png"):
    """
    Plots the training losses and saves the plot to a file.
    
    Args:
        total_loss_history (list): List of average total loss per epoch.
        moment_loss_history (list): List of average Moment-DETR loss per epoch.
        yt_loss_history (list): List of average YouTube Highlight loss per epoch.
        save_path (str): File path to save the plot.
    """
    fig, ax_loss = plt.subplots(1, 1, figsize=(12, 9))
    ax_loss.plot(epochs_list, total_loss_history, label="Total Loss")
    ax_loss.plot(epochs_list, moment_loss_history, label="Moment-DETR Loss")
    ax_loss.plot(epochs_list, yt_loss_history, label="YouTube Highlight Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log10)")
    ax_loss.set_title("Training/Validation Loss")
    ax_loss.legend()
    ax_loss.set_xlim([0, epochs_list[-1]+1])
    #ax_loss.set_ylim([0, 1])
    ax_loss.grid(True, which="both", linestyle='--', alpha=0.75,  color='gray')
    plt.savefig(save_path, bbox_inches="tight",
            pad_inches=0.3, transparent=True, dpi=300)



def ema(values, alpha=0.2):
    """Exponential Moving Average smoothing."""
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def plot_and_save_losses_smoothed(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot_smoothed.png", alpha_ema=0.2):
    """
    Plots the smoothed training losses using EMA and saves the plot to a file.
    
    Args:
        total_loss_history (list): List of average total loss per epoch.
        moment_loss_history (list): List of average Moment-DETR loss per epoch.
        yt_loss_history (list): List of average YouTube Highlight loss per epoch.
        save_path (str): File path to save the plot.
        alpha (float): Smoothing factor for EMA.
    """
    total_smooth = ema(total_loss_history, alpha_ema)
    moment_smooth = ema(moment_loss_history, alpha_ema)
    yt_smooth = ema(yt_loss_history, alpha_ema)

    fig, ax_loss = plt.subplots(1, 1, figsize=(12, 9))
    ax_loss.plot(epochs_list, total_smooth, label="Total Loss (EMA)")
    ax_loss.plot(epochs_list, moment_smooth, label="Moment-DETR Loss (EMA)")
    ax_loss.plot(epochs_list, yt_smooth, label="YouTube Highlight Loss (EMA)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log10)")
    ax_loss.set_title("Training/Validation Loss (Smoothed)")
    ax_loss.legend()
    ax_loss.set_xlim([0, epochs_list[-1]+1])
    ax_loss.grid(True, which="both", linestyle='--', alpha=0.75, color='gray')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, transparent=True, dpi=300)
    

def plot_and_save_losses_all(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot_all.png", alpha_ema=0.2):
    """
    Plots both raw and smoothed (EMA) training losses and saves the plot to a file.

    Args:
        total_loss_history (list): List of average total loss per epoch.
        moment_loss_history (list): List of average Moment-DETR loss per epoch.
        yt_loss_history (list): List of average YouTube Highlight loss per epoch.
        save_path (str): File path to save the plot.
        alpha (float): Smoothing factor for EMA.
    """
    # Smoothed with EMA
    total_smooth = ema(total_loss_history, alpha_ema)
    moment_smooth = ema(moment_loss_history, alpha_ema)
    yt_smooth = ema(yt_loss_history, alpha_ema)

    fig, ax_loss = plt.subplots(1, 1, figsize=(12, 9))

    # Plot raw losses (with lower opacity)
    ax_loss.plot(epochs_list, total_loss_history, label="Total Loss (Raw)", alpha=0.25)
    ax_loss.plot(epochs_list, moment_loss_history, label="Moment-DETR Loss (Raw)", alpha=0.25)
    ax_loss.plot(epochs_list, yt_loss_history, label="YouTube Highlight Loss (Raw)", alpha=0.25)

    # Plot smoothed losses
    ax_loss.plot(epochs_list, total_smooth, label="Total Loss (EMA)", linewidth=3)
    ax_loss.plot(epochs_list, moment_smooth, label="Moment-DETR Loss (EMA)", linewidth=3)
    ax_loss.plot(epochs_list, yt_smooth, label="YouTube Highlight Loss (EMA)", linewidth=3)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log10)")
    ax_loss.set_title("Training/Validation Loss (Raw + Smoothed)")
    ax_loss.legend()
    ax_loss.set_xlim([0, epochs_list[-1]+1])
    ax_loss.grid(True, which="both", linestyle='--', alpha=0.75, color='gray')

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, transparent=True, dpi=300)

def train(model, dataloader, args):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(args.min_lr / args.lr) ** (1 / args.num_epochs)
    )
    total_loss_history, moment_loss_history, yt_loss_history = [], [], []
    for epoch in range(args.num_epochs):
        for i_batch, sample_batched in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            durations = [x['duration'] for x in sample_batched]
            
            new_batch_in_75 = process_batch(sample_batched)
            feature_in_75 = new_batch_in_75['feature'].to(args.device)
            moment_detr_raw_in_75 = new_batch_in_75['moment_detr_saliency_raw'].to(args.device).unsqueeze(-1)
            
            new_batch_in_max_length = process_batch_2(sample_batched)
            feature_in_max_length = new_batch_in_max_length['feature'].to(args.device)
            yt_highlite_raw_in_max_length = new_batch_in_max_length['yt_highlite_saliency_raw'].to(args.device).unsqueeze(-1)
            # yt_highlite_raw_in_max_length = [x[:d] - 0.5 for x, d in zip(yt_highlite_raw_in_max_length, durations)]
            yt_highlite_normalized = []
            # yt_pow = 1.0
            yt_pow = 0.2
            for h in yt_highlite_raw_in_max_length:
                h = h ** yt_pow
                mean = h.mean()
                std = h.std()
                normalized = (h - mean) / (std + 1e-8)  # add epsilon to avoid division by zero
                yt_highlite_normalized.append(normalized)
            
            predicted_saliency_raw_in_75 = model(feature_in_75)
            
            norm = nn.Identity()
            predicted_saliency_raw_in_max_length = model(feature_in_max_length)
            # predicted_saliency_raw_in_max_length = [norm(x[:d]) for x, d in zip(predicted_saliency_raw_in_max_length, durations)]
            predicted_saliency_normalized = []
            for h in predicted_saliency_raw_in_max_length:
                mean = h.mean()
                std = h.std()
                normalized = (h - mean) / (std + 1e-8)  # add epsilon to avoid division by zero
                predicted_saliency_normalized.append(normalized)
            
            loss_moment_detr = criterion(predicted_saliency_raw_in_75, moment_detr_raw_in_75)
            
            loss_yt_highlite = [criterion(x, y) for x, y in zip(predicted_saliency_raw_in_max_length, yt_highlite_raw_in_max_length)]
            loss_yt_highlite = sum(loss_yt_highlite) / len(loss_yt_highlite)
            
            alpha = 0.3
            beta = 1.0
            loss_moment_detr = alpha * loss_moment_detr
            loss_yt_highlite = beta * loss_yt_highlite
            
            loss = loss_moment_detr + loss_yt_highlite
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        # add loss logscale
        eps = 1e-8
        total_loss_history.append(np.log10(train_loss + eps))
        moment_loss_history.append(np.log10(loss_moment_detr.item() + eps))
        yt_loss_history.append(np.log10(loss_yt_highlite.item() + eps))
        # total_loss_history.append(train_loss)
        # moment_loss_history.append(loss_moment_detr.item())
        # yt_loss_history.append(loss_yt_highlite.item())
        if epoch % 10 == 0:
            # Use the current loss as the train loss (you may average over the epoch if desired)
            msg = f"    Epoch [{epoch+1}/{args.num_epochs}]| Total Loss={train_loss:.4f} | Moment-DETR Loss={loss_moment_detr:.4f} | Youtube Highlight Loss={loss_yt_highlite:.4f}"
            msg  += f" | LR={scheduler.get_last_lr()[0]:.4f}"
            epochs_list = list(range(epoch+1))
            plot_and_save_losses(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot_"+args.model_type+".png")
            plot_and_save_losses_smoothed(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot_smoothed_"+args.model_type+".png", alpha_ema=0.4)
            plot_and_save_losses_all(epochs_list, total_loss_history, moment_loss_history, yt_loss_history, save_path="loss_plot_all_"+args.model_type+".png", alpha_ema=0.4)
            print(msg)
        scheduler.step()
        
        if epoch % args.save_every == 0 and epoch != 0:
            # epoch to be saved in name XXXXX 5 digit, zerofill
            torch.save(model, f'ckpt/{args.model_type}_e{epoch:05d}.pt')
    
    torch.save(model, f'ckpt/{args.model_type}_e{epoch:05d}.pt')

# Define a named collate function instead of using a lambda.
def collate_batch(batch):
    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--feature_dir', type=str, default='output_features')
    parser.add_argument('--annotation_dir', type=str, default='annotations')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='TransformerNet')
    parser.add_argument('--moment_detr_ckpt', type=str, default=None)
    parser.add_argument('--num_sa_blocks', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda') # 'cpu', 'cuda' or 'mps' for mac
    args = parser.parse_args()
    
    args.device = torch.device(args.device)  # Convert to torch.device object
    # args.moment_detr_ckpt = './ckpt/moment_detr/ft_model_from_pt_model_e50.ckpt'
    
    # Allow argparse.Namespace as a safe global for weights_only loading. This may be dependent on PyTorch version.
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
        if args.moment_detr_ckpt is not None:
            weight = torch.load(args.moment_detr_ckpt, map_location=args.device, weights_only=True)
            saliency_proj = nn.Linear(256, 1).to(args.device)
            saliency_proj.weight = nn.Parameter(weight['model']['saliency_proj.weight'])
            saliency_proj.bias = nn.Parameter(weight['model']['saliency_proj.bias'])
        else:
            saliency_proj = nn.Linear(256, 1).to(args.device)  # Random initialization
    except:
        if args.moment_detr_ckpt is not None:
            weight = torch.load(args.moment_detr_ckpt, map_location=args.device)
            saliency_proj = nn.Linear(256, 1).to(args.device)
            saliency_proj.weight = nn.Parameter(weight['model']['saliency_proj.weight'])
            saliency_proj.bias = nn.Parameter(weight['model']['saliency_proj.bias'])
        else:
            saliency_proj = nn.Linear(256, 1).to(args.device)  # Random initialization
        
    if args.model_type == 'SimpleConv':
        model = SimpleConv(saliency_proj).to(args.device)
    elif args.model_type == 'SANet':
        model = SANet(saliency_proj, args.num_sa_blocks).to(args.device)
    elif args.model_type == 'TransformerNet':
        model = TransformerNet(saliency_proj, args.num_sa_blocks).to(args.device)
    else:
        raise ValueError('Invalid model type')
    
    dataset = SaliencyDataset(args.root_dir, args.feature_dir, args.annotation_dir)
    # Replace the lambda with our named collate function
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_batch
    )
    
    train(model, dataloader, args)
