import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import math

from timm.data.random_erasing import RandomErasing
from utility import RandomIdentitySampler,RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

__factory = {
    'Mars':Mars,
    'iLIDSVID':iLIDSVID,
    'PRID':PRID
}


def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a deterministic seed.
    PyTorch sets torch.initial_seed() for each worker; we align numpy and python random with it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_collate_fn(batch):
    
    
    imgs, pids, camids,a= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(a, dim=0)

def val_collate_fn(batch):
    
    imgs, pids, camids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch,  img_paths



def pair_collate_fn(batch):
    v1, v2, labels = zip(*batch)
    v1 = torch.stack(v1, dim=0)
    v2 = torch.stack(v2, dim=0)
    labels = torch.stack(labels, dim=0)
    return v1, v2, labels

def dataloader(Dataset_name):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
            
        ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=4, sample='intelligent',transform=train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

   
    train_loader = DataLoader(train_set, batch_size=64,sampler=RandomIdentitySampler(dataset.train, 64,4),num_workers=4, collate_fn=train_collate_fn)
  
    q_val_set = VideoDataset(dataset.query, seq_len=4, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=4, sample='dense', transform=val_transforms)
    
    
    return train_loader, len(dataset.query), num_classes, cam_num, view_num,q_val_set,g_val_set



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoPairDataset(Dataset):
    """Video Pair Dataset for Cross-Video Matching.
    Returns (video1_tensor, video2_tensor, label) where label in {1.0, 0.0}.
    Each video tensor has shape [seq_len, C, H, W].
    """
    def __init__(self, dataset, seq_len=8, transform=None):
        # dataset: list of (img_paths, pid, camid)
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform

        # group indices by pid -> list of tracklet indices
        self.pid_to_indices = {}
        for index, (_, pid, _) in enumerate(self.dataset):
            if pid in self.pid_to_indices:
                self.pid_to_indices[pid].append(index)
            else:
                self.pid_to_indices[pid] = [index]
        self.pids = list(self.pid_to_indices.keys())

    def __len__(self):
        # one epoch length aligned with number of tracklets for convenience
        return len(self.dataset)

    def _sample_video_frames(self, img_paths):
        """Sample a clip of length seq_len from img_paths and apply transform.
        Strategy matches VideoDataset 'random' sampling.
        """
        num = len(img_paths)
        frame_indices = range(num)
        rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.seq_len, len(frame_indices))
        indices = frame_indices[begin_index:end_index]
        if len(indices) < self.seq_len:
            indices = list(indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            indices = list(indices)

        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)  # [seq_len, C, H, W]
        return imgs

    def _sample_video_frames_with_begin(self, img_paths, begin_index: int):
        """Sample a clip with a specified begin index, used to enforce distinct clips."""
        num = len(img_paths)
        frame_indices = range(num)
        begin_index = max(0, min(begin_index, max(0, len(frame_indices) - 1)))
        end_index = min(begin_index + self.seq_len, len(frame_indices))
        indices = frame_indices[begin_index:end_index]
        if len(indices) < self.seq_len:
            indices = list(indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            indices = list(indices)

        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def __getitem__(self, _index):
        # decide positive (1.0) or negative (0.0) with 50% probability
        is_positive = random.random() < 0.5

        if is_positive:
            # pick a pid with at least one tracklet
            pid = random.choice(self.pids)
            idxs = self.pid_to_indices[pid]
            if len(idxs) >= 2:
                i1, i2 = random.sample(idxs, 2)
            else:
                i1 = i2 = idxs[0]
            (img_paths1, _, _ ) = self.dataset[i1]
            if i1 != i2:
                (img_paths2, _, _ ) = self.dataset[i2]
                v1 = self._sample_video_frames(img_paths1)
                v2 = self._sample_video_frames(img_paths2)
            else:
                # same tracklet: take two distinct clips by enforcing different begin indices when possible
                num = len(img_paths1)
                if num > self.seq_len + 1:
                    rand_end = max(0, num - self.seq_len - 1)
                    begin1 = random.randint(0, rand_end)
                    begin2 = random.randint(0, rand_end)
                    if begin2 == begin1 and rand_end > 0:
                        begin2 = (begin1 + self.seq_len) % (rand_end + 1)
                    v1 = self._sample_video_frames_with_begin(img_paths1, begin1)
                    v2 = self._sample_video_frames_with_begin(img_paths1, begin2)
                else:
                    v1 = self._sample_video_frames(img_paths1)
                    v2 = self._sample_video_frames(img_paths1)
            label = torch.tensor(1.0, dtype=torch.float32)
            return v1, v2, label
        else:
            # pick two different pids
            if len(self.pids) >= 2:
                pid1, pid2 = random.sample(self.pids, 2)
            else:
                # degenerate case: fall back to same pid if only one exists
                pid1 = pid2 = self.pids[0]
            i1 = random.choice(self.pid_to_indices[pid1])
            i2 = random.choice(self.pid_to_indices[pid2])
            (img_paths1, _, _ ) = self.dataset[i1]
            (img_paths2, _, _ ) = self.dataset[i2]
            v1 = self._sample_video_frames(img_paths1)
            v2 = self._sample_video_frames(img_paths2)
            label = torch.tensor(0.0, dtype=torch.float32)
            return v1, v2, label

class QueryGalleryPairDataset(Dataset):
    """Validation Pair Dataset built from Query and Gallery sets.
    - Deterministically builds a balanced list of positive/negative pairs.
    - Uses centered sampling for stability across epochs.
    Returns (video1_tensor, video2_tensor, label) where label in {1.0, 0.0}.
    Each video tensor has shape [seq_len, C, H, W].
    """
    def __init__(self, query_set, gallery_set, seq_len=8, transform=None):
        self.query = query_set
        self.gallery = gallery_set
        self.seq_len = seq_len
        self.transform = transform

        # Build pid->indices maps
        self.q_pid_to_indices = {}
        for idx, (_, pid, _) in enumerate(self.query):
            self.q_pid_to_indices.setdefault(pid, []).append(idx)
        self.g_pid_to_indices = {}
        for idx, (_, pid, _) in enumerate(self.gallery):
            self.g_pid_to_indices.setdefault(pid, []).append(idx)

        # Build deterministic positive pairs: one gallery per query if exists
        self.pairs = []  # (q_idx, g_idx, label)
        pos_q_indices = []
        for q_idx, (_, q_pid, _) in enumerate(self.query):
            g_list = self.g_pid_to_indices.get(q_pid, [])
            if g_list:
                g_idx = g_list[0]
                self.pairs.append((q_idx, g_idx, 1.0))
                pos_q_indices.append(q_idx)

        # Build negative pairs: one per positive query with a different pid
        gallery_pids = list(self.g_pid_to_indices.keys())
        for q_idx in pos_q_indices:
            _, q_pid, _ = self.query[q_idx]
            neg_pid = None
            for pid in gallery_pids:
                if pid != q_pid:
                    neg_pid = pid
                    break
            if neg_pid is None:
                # fallback: skip if gallery has only one pid
                continue
            g_idx = self.g_pid_to_indices[neg_pid][0]
            self.pairs.append((q_idx, g_idx, 0.0))

    def __len__(self):
        return len(self.pairs)

    def _sample_video_frames_center(self, img_paths):
        """Deterministic, centered sampling for validation stability."""
        num = len(img_paths)
        frame_indices = range(num)
        if num <= self.seq_len:
            indices = list(frame_indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            begin_index = (num - self.seq_len) // 2
            end_index = begin_index + self.seq_len
            indices = list(frame_indices)[begin_index:end_index]
        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def __getitem__(self, index):
        q_idx, g_idx, label = self.pairs[index]
        (q_img_paths, _, _) = self.query[q_idx]
        (g_img_paths, _, _) = self.gallery[g_idx]
        v1 = self._sample_video_frames_center(q_img_paths)
        v2 = self._sample_video_frames_center(g_img_paths)
        label = torch.tensor(label, dtype=torch.float32)
        return v1, v2, label

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            targt_cam=[]
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                targt_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            targt_cam=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    targt_cam.append(camid)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, targt_cam,img_paths
            #return imgs_array, pid, int(camid),trackid


        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

            

        
class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        targt_cam=[]
        
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
            targt_cam.append(camid)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        
        return imgs, pid, targt_cam ,labels
        


def make_pair_dataloader(Dataset_name, seq_len=4, batch_size=32, num_workers=4, seed: int = 42):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = __factory[Dataset_name]()

    train_pair_set = VideoPairDataset(dataset.train, seq_len=seq_len, transform=train_transforms)
    val_pair_set = QueryGalleryPairDataset(dataset.query, dataset.gallery, seq_len=seq_len, transform=val_transforms)

    # Deterministic generator for shuffle and transforms
    g = torch.Generator()
    g.manual_seed(seed)

    train_pair_loader = DataLoader(
        train_pair_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pair_collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_pair_loader = DataLoader(
        val_pair_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pair_collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_pair_loader, val_pair_loader