import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import json
import torch.nn.functional as F

class SaliencyDataset(Dataset):
    def __init__(self, root_dir, feature_dir, annotation_dir, transform=None, sampled_freq=2):
        self.root_dir = root_dir
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.sampled_freq = sampled_freq
        
        self.feature_dir = os.path.join(self.root_dir, self.feature_dir)
        self.annotation_paths = os.listdir(os.path.join(self.root_dir, self.annotation_dir))
        
        self.data = []
        for annotation_path in self.annotation_paths:
            with open(os.path.join(self.root_dir, self.annotation_dir, annotation_path)) as f:
                self.data.extend([json.loads(line) for line in f])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        
        sample = {}
        sample['video_id'] = data_sample['video_id']
        sample['moment_detr_saliency'] = torch.tensor(data_sample['predicted_saliency'])
        sample['moment_detr_saliency_raw'] = torch.tensor(data_sample['full_saliency'])
        sample['yt_highlite_saliency'] = torch.tensor(data_sample['yutube_highlight'])
        sample['yt_highlite_saliency_raw'] = torch.tensor(data_sample['full_youtube_intensity'])
        sample['video_name'] = data_sample['query']
        sample['url'] = data_sample['url']
        
        sample['feature'] = torch.load(os.path.join(self.feature_dir, 'Features_' + f'{data_sample["video_id"]}.pt'))
        
        sample['duration'] = sample['feature'].shape[1]
        
        return sample

import torch
import torch.nn.functional as F

def process_batch(sample_batched):
    processed_features = []
    processed_moment_detr_raws = []
    processed_yt_highlite_raws = []
    processed_moment_detr_saliency = []
    processed_yt_highlite_saliency = []

    for i in range(len(sample_batched)):
        feature = sample_batched[i]['feature']  # Shape: [1, seq_len, 256]
        moment_detr_raws = sample_batched[i]['moment_detr_saliency_raw'].unsqueeze(0)  # [1, seq_len]
        yt_highlite_raws = sample_batched[i]['yt_highlite_saliency_raw'].unsqueeze(0)  # [1, seq_len]
        moment_detr_saliency = sample_batched[i]['moment_detr_saliency'].unsqueeze(0)  # [1, seq_len]
        yt_highlite_saliency = sample_batched[i]['yt_highlite_saliency'].unsqueeze(0)  # [1, seq_len]

        # in some reason the length of feature and saliency may not be the same
        # so we need to make them the same length based on the minimum length
        min_length = min(feature.shape[1], moment_detr_raws.shape[1], yt_highlite_raws.shape[1], moment_detr_saliency.shape[1], yt_highlite_saliency.shape[1])
        feature = feature[:, :min_length, :]
        moment_detr_raws = moment_detr_raws[:, :min_length]
        yt_highlite_raws = yt_highlite_raws[:, :min_length]
        moment_detr_saliency = moment_detr_saliency[:, :min_length]
        yt_highlite_saliency = yt_highlite_saliency[:, :min_length]
        
        # Split each tensor into chunks of 75
        feature_chunks = torch.split(feature, 75, dim=1)
        moment_detr_raws_chunks = torch.split(moment_detr_raws, 75, dim=1)
        yt_highlite_raws_chunks = torch.split(yt_highlite_raws, 75, dim=1)
        moment_detr_saliency_chunks = torch.split(moment_detr_saliency, 75, dim=1)
        yt_highlite_saliency_chunks = torch.split(yt_highlite_saliency, 75, dim=1)
        
        # Process each chunk
        for j in range(len(feature_chunks)):
            chunk_f = feature_chunks[j]
            chunk_md_raw = moment_detr_raws_chunks[j]
            chunk_yt_raw = yt_highlite_raws_chunks[j]
            chunk_md_sal = moment_detr_saliency_chunks[j]
            chunk_yt_sal = yt_highlite_saliency_chunks[j]

            # If chunk is shorter than 75, pad it
            if chunk_f.shape[1] < 75:
                pad_size = 75 - chunk_f.shape[1]
                chunk_f = F.pad(chunk_f, (0, 0, 0, pad_size))  # Pad along dim=1
                chunk_md_raw = F.pad(chunk_md_raw, (0, pad_size))
                chunk_yt_raw = F.pad(chunk_yt_raw, (0, pad_size))
                chunk_md_sal = F.pad(chunk_md_sal, (0, pad_size))
                chunk_yt_sal = F.pad(chunk_yt_sal, (0, pad_size))

            processed_features.append(chunk_f)
            processed_moment_detr_raws.append(chunk_md_raw)
            processed_yt_highlite_raws.append(chunk_yt_raw)
            processed_moment_detr_saliency.append(chunk_md_sal)
            processed_yt_highlite_saliency.append(chunk_yt_sal)

    # Concatenate all processed tensors along the batch dimension
    new_batch = {
        'feature': torch.cat(processed_features, dim=0),  # Shape: (n, 75, 256)
        'moment_detr_saliency_raw': torch.cat(processed_moment_detr_raws, dim=0),  # Shape: (n, 75, 1)
        'yt_highlite_saliency_raw': torch.cat(processed_yt_highlite_raws, dim=0),  # Shape: (n, 75, 1)
        'moment_detr_saliency': torch.cat(processed_moment_detr_saliency, dim=0),  # Shape: (n, 75, 1)
        'yt_highlite_saliency': torch.cat(processed_yt_highlite_saliency, dim=0)  # Shape: (n, 75, 1)
    }

    return new_batch


def process_batch_2(sample_batched, pad_value=0):
    batch = sample_batched.copy()
    max_len = max(x['yt_highlite_saliency_raw'].shape[0] for x in batch)  # Find max length in second dim
    
    for i in range(len(batch)):
        feature = batch[i]['feature']  # Shape: [1, seq_len, 256]
        moment_detr_raws = batch[i]['moment_detr_saliency_raw'].unsqueeze(0)  # [1, seq_len]
        yt_highlite_raws = batch[i]['yt_highlite_saliency_raw'].unsqueeze(0)  # [1, seq_len]
        moment_detr_saliency = batch[i]['moment_detr_saliency'].unsqueeze(0)  # [1, seq_len]
        yt_highlite_saliency = batch[i]['yt_highlite_saliency'].unsqueeze(0)  # [1, seq_len]

        # in some reason the length of feature and saliency may not be the same
        # so we need to make them the same length based on the minimum length
        min_length = min(feature.shape[1], moment_detr_raws.shape[1], yt_highlite_raws.shape[1], moment_detr_saliency.shape[1], yt_highlite_saliency.shape[1])
        feature = feature[:, :min_length, :]
        moment_detr_raws = moment_detr_raws[:, :min_length]
        yt_highlite_raws = yt_highlite_raws[:, :min_length]
        moment_detr_saliency = moment_detr_saliency[:, :min_length]
        yt_highlite_saliency = yt_highlite_saliency[:, :min_length]
        
        pad_size = max_len - feature.shape[1]  # Compute padding amount
        
        # Pad with zeros along the second dimension
        padded_feature = torch.cat(
            [feature, torch.full((1, pad_size, feature.shape[2]), pad_value, dtype=feature.dtype, device=feature.device)],
            dim=1
        )
        padded_detr_raws = torch.cat(
            [moment_detr_raws, torch.full((1, pad_size), pad_value, dtype=moment_detr_raws.dtype, device=moment_detr_raws.device)],
            dim=1
        )
        padded_yt_raws = torch.cat(
            [yt_highlite_raws, torch.full((1, pad_size), pad_value, dtype=yt_highlite_raws.dtype, device=yt_highlite_raws.device)],
            dim=1
        )
        padded_detr_saliency = torch.cat(
            [moment_detr_saliency, torch.full((1, pad_size), pad_value, dtype=moment_detr_saliency.dtype, device=moment_detr_saliency.device)],
            dim=1
        )
        padded_yt_saliency = torch.cat(
            [yt_highlite_saliency, torch.full((1, pad_size), pad_value, dtype=yt_highlite_saliency.dtype, device=yt_highlite_saliency.device)],
            dim=1
        )
        
        
        batch[i]['feature'] = padded_feature  # Replace with padded tensor
        batch[i]['moment_detr_saliency_raw'] = padded_detr_raws
        batch[i]['yt_highlite_saliency_raw'] = padded_yt_raws
        batch[i]['moment_detr_saliency'] = padded_detr_saliency
        batch[i]['yt_highlite_saliency'] = padded_yt_saliency
    
    new_batch = {}
    for key in ['feature', 'moment_detr_saliency_raw', 'yt_highlite_saliency_raw', 'moment_detr_saliency', 'yt_highlite_saliency']:
        new_batch[key] = torch.stack([x[key] for x in batch], dim=0).squeeze()

    return new_batch


if __name__ == '__main__':
    root_dir = 'data'
    feature_dir = 'output_features'
    annotation_dir = 'annotations'
    sampled_freq = 2
    dataset = SaliencyDataset(root_dir, feature_dir, annotation_dir, sampled_freq)
    print(len(dataset))
    print(dataset[0])
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda batch: batch)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0]['video_id'])
        new_batch = process_batch(sample_batched)
        new_batch2 = process_batch_2(sample_batched)
        break   