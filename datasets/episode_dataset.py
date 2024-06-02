import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import random

train_cats = [
    'hair drier', 'clock', 'wine glass', 'book', 'cake', 'tie', 'motorcycle',
    'sheep', 'bottle', 'giraffe', 'cell phone', 'suitcase', 'remote', 'bench',
    'mouse', 'carrot', 'banana', 'train', 'sports ball', 'toothbrush', 'fire hydrant',
    'airplane', 'tv', 'bus', 'refrigerator', 'couch', 'knife', 'toilet', 'elephant',
    'truck', 'parking meter', 'car', 'potted plant', 'kite', 'skateboard', 'orange',
    'horse', 'cat', 'tennis racket', 'bowl', 'scissors', 'baseball glove', 'apple',
    'traffic light', 'handbag', 'donut', 'dog', 'hot dog', 'oven', 'umbrella', 'sink',
    'pizza'
]
val_cats = [
    'cow', 'dining table', 'zebra', 'sandwich', 'bear', 'toaster', 'person',
    'laptop', 'bed', 'teddy bear', 'baseball bat', 'skis'
]
test_cats = [
    'bicycle', 'boat', 'stop sign', 'bird', 'backpack', 'frisbee', 'snowboard',
    'surfboard', 'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave',
    'vase'
]


class Dataset(torch.utils.data.Dataset):

    def __init__(self, phase, coco_jsons, data_dir, augment, repeats = 1, sub_batch_size = 4):
        super(Dataset, self).__init__()
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        # Prepare image_paths and labels
        cats = {'train': train_cats, 'val': val_cats, 'test': test_cats}[phase]

        idx2cat = list(cats)
        self.idx2cat = idx2cat
        cat2idx = dict([(cat, idx) for idx, cat in enumerate(idx2cat)])
        self.cat2idx = cat2idx
        # to construct episode_based data setting
        self.cats = cats
        self.sub_cats = {}
        for each_cat in self.cats:
            self.sub_cats[cat2idx[each_cat]] = []
        image_paths = []
        labels = []
        image_to_label = dict()
        for coco_json in coco_jsons:
            with open(coco_json, 'r') as reader:
                data_info = eval(reader.read())
            for image_name in data_info.keys():
                if set(data_info[image_name]['categories']) - set(cats):
                    continue
                label = torch.zeros(1, len(cats), dtype=torch.float)
                image_path = data_dir + '/' + image_name.split('_')[1] + '/' + image_name
                for cat in data_info[image_name]['categories']:
                    label[0, cat2idx[cat]] = 1
                    self.sub_cats[cat2idx[cat]].append(image_path)
                image_paths.append(image_path)
                labels.append(label)
                image_to_label[image_path] = label
        self.image_paths = image_paths
        self.labels = torch.cat(labels, dim=0)
        self.image_to_label = image_to_label

        aug = []
        aug += [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.aug_func = transforms.Compose(aug)

        # Build sub_dataloader
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = sub_batch_size, shuffle = True, num_workers = 0, pin_memory = False)
        for each_cat in self.cats:
            sub_dataset = SubDataset(self.sub_cats[cat2idx[each_cat]], self.image_to_label, each_cat, self.aug_func)
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))

        self.repeats = repeats
        self.info = (
            f'{phase} dataset:\n'
            f'From coco_jsons: {coco_jsons}\n'
            f'There are {len(self.image_paths)} images and {len(cats)} classes, with {repeats} repeats\n'
            f'Augmentations are {aug}\n'
            f'Label counts:\n'
            f'{idx2cat}\n'
            f'{torch.sum(self.labels, dim=0).numpy().tolist()}\n'
        )

    def __len__(self):
        return len(self.cats) * self.repeats

    def __getitem__(self, index):
        return next(iter(self.sub_dataloader[index % len(self.cats)]))

class SubDataset(torch.utils.data.Dataset):
    def __init__(self, sub_cats, image_to_label, cat_name, aug_func):
        super(SubDataset, self).__init__()
        self.sub_cats = sub_cats
        self.image_to_label = image_to_label
        self.aug_func = aug_func
        self.cat_name = cat_name
        self.sub_cats_num = len(self.sub_cats)

    def __len__(self):
        return len(self.sub_cats)

    def __getitem__(self, idx):
        t_idx = random.randint(0, self.sub_cats_num - 1)
        image = Image.open(self.sub_cats[t_idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.aug_func(image)
        label = self.image_to_label[self.sub_cats[t_idx]]
        return image, label, self.cat_name, False, False
