import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
# import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

import datasets
from metrics import Metric


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
        phase, epoch, configs, data_loader, model, optimizer,
        lr_schedule, metric, logger, logger_all
    ):
    end = time.time()
    total_scores = {}
    logger.info(f'Epoch {epoch} with {len(data_loader)} iterations')
    if phase != 'train':
        all_pred_scores = []
        all_gt_labels = []
    # Collect all of the predicting labels
    for b_ind, (images, gt_labels, indexs, lstm_labels, label_numbers) in enumerate(data_loader):
        images = images.cuda(non_blocking = True)
        gt_labels = gt_labels.cuda(non_blocking = True)
        if phase == 'train':
            # update learning rate
            iteration = epoch * len(data_loader) + b_ind
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
            # forward
            if not configs.meta_train:
                loss_names, loss_items, total_loss = model(images, gt_labels)
            else:
                loss_names, loss_items, total_loss = model.module.meta_train(images, gt_labels)
            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # record losses
            scores = {}
            for loss_name, loss_item in zip(loss_names, loss_items):
                scores[loss_name] = loss_item
        else:
            # forward
            with torch.no_grad():
                if 'LSTM' in configs.model.kind:
                    pred_scores, gt_labels = model.module.infer(images, lstm_labels, label_numbers)

                else:
                    pred_scores = model.module.infer(images)
                    pred_scores = pred_scores.cpu().numpy()
                    gt_labels = gt_labels.cpu().numpy()

            all_pred_scores.append(pred_scores)
            all_gt_labels.append(gt_labels)
            scores = {}
        for key, value in scores.items():
            if not key in total_scores:
                total_scores[key] = []
            total_scores[key].append(value)
        if configs.device.type == 'cuda':
            memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        else:
            memory = 0.
        time_cost = time.time() - end
        logger_all.info(
            f'{configs.tag} {phase} '
            f'Rank {dist.get_rank()}/{dist.get_world_size()} '
            f'Epoch {epoch}/{configs.max_epoch} '
            f'Batch {b_ind}/{len(data_loader)} '
            f'Time {time_cost: .3f} Mem {memory}MB '
            f'LR {optimizer.param_groups[0]["lr"]:.3e} '
            f'{scores}'
        )
    if phase != 'train':
        all_pred_scores = np.concatenate(all_pred_scores, axis=0)
        all_gt_labels = np.concatenate(all_gt_labels, axis=0)
        print(all_pred_scores.shape, all_gt_labels.shape)
        total_scores = metric(all_pred_scores, all_gt_labels)
    else:
        total_scores = dict(
            [
                (key, round(np.mean(value), 4))
                for key, value in total_scores.items()
            ]
        )
    return total_scores

def run_episode(configs, model, data_loader, metric, logger, logger_all):
    micro_ap = 0
    macro_ap = 0
    mi_prec = 0
    ma_prec = 0
    mi_recall = 0
    ma_recall = 0
    mi_f1 = 0
    ma_f1 = 0
    all_preds = []
    all_gts = []
    for b_ind, (images, gt_labels, indexs, lstm_labels, label_numbers) in enumerate(data_loader):
        images = images.cuda(non_blocking = True)
        gt_labels = gt_labels.cuda(non_blocking = True)
        with torch.no_grad():
            pred_scores, gt_labels = model.module.infer(images, gt_labels)
            pred_scores = pred_scores.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()
            all_preds.append(pred_scores)
            all_gts.append(gt_labels)
            total_scores = metric(pred_scores, gt_labels)
            micro_ap += total_scores['micro_ap']
            macro_ap += total_scores['macro_ap']
            mi_prec += total_scores['mi_prec']
            ma_prec += total_scores['ma_prec']
            mi_recall += total_scores['mi_recall']
            ma_recall += total_scores['ma_recall']
            mi_f1 += total_scores['mi_f1']
            ma_f1 += total_scores['ma_f1']
        logger_all.info(f'Batch {b_ind}/{len(data_loader)} ')

    micro_ap = micro_ap / len(data_loader)
    macro_ap = macro_ap / len(data_loader)
    mi_prec = mi_prec / len(data_loader)
    ma_prec = ma_prec / len(data_loader)
    mi_recall = mi_recall / len(data_loader)
    ma_recall = ma_recall / len(data_loader)
    mi_f1 = mi_f1 / len(data_loader)
    ma_f1 = ma_f1 / len(data_loader)

    return {'micro_ap': round(micro_ap * 100, 2), 'macro_ap': round(macro_ap * 100, 2),
            'mi_prec': round(mi_prec * 100, 2), 'ma_prec': round(ma_prec * 100, 2),
            'mi_recall': round(mi_recall * 100, 2), 'ma_recall': round(ma_recall * 100, 2),
            'mi_f1': round(mi_f1 * 100, 2), 'ma_f1': round(ma_f1 * 100, 2)}

def prepare_prototypes(configs, data_loader, model, logger, logger_all):
    logger.info(f'Generate Multi-label Prototypes')
    assert len(data_loader) == 1
    prototypes = None
    for b_ind, (images, gt_labels, indexs, lstm_labels, label_numbers) in enumerate(data_loader):
        images = images.cuda(non_blocking = True)
        gt_labels = gt_labels.cuda(non_blocking = True)
        with torch.no_grad():
            model.module.generate_prototypes(images, gt_labels)

    return

def main(configs, is_test, model, optimizer, logger, logger_all, **kwargs):
    # Get metric function
    metric = Metric(**configs.metric)
    # Build dataloader
    d_kind = configs.dataset.k_kind if 'k_kind' in configs.dataset else configs.dataset.kind
    train_dataset = datasets.__dict__[d_kind](**configs.dataset.kwargs)
    logger.info(train_dataset.info)
    shuffle = True if 'shuffle' not in configs.dataset else configs.dataset.shuffle
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=shuffle
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=configs.dataset.train_batch_size,
        num_workers=configs.dataset.num_workers,
        pin_memory=False, drop_last=True,
        worker_init_fn=seed_worker
    )
    logger.info(f'{len(train_dataset)} train images')
    lr_configs = configs.trainer.lr_schedule
    warmup_lr_schedule = np.linspace(
        lr_configs.start_warmup, lr_configs.base_lr,
        len(train_loader) * lr_configs.warmup_epochs
    )
    iters = np.arange(len(train_loader) * lr_configs.cosine_epochs)
    cosine_lr_schedule = np.array(
        [
            lr_configs.final_lr + (
                0.5 * (lr_configs.base_lr - lr_configs.final_lr)
                * (1 + math.cos(math.pi * t / (len(train_loader) * lr_configs.cosine_epochs)))
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate([warmup_lr_schedule] + [cosine_lr_schedule] * lr_configs.cosine_times)
    plt.plot(range(len(lr_schedule)), lr_schedule)
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.savefig(configs.metric_dir + '/lr.jpg')
    configs.max_epoch = lr_configs.warmup_epochs + lr_configs.cosine_epochs * lr_configs.cosine_times

    for epoch in range(configs.max_epoch):
        dist.barrier()
        train_sampler.set_epoch(epoch)
        model.train()
        score = run_epoch(
            phase='train', epoch=epoch, configs=configs, data_loader=train_loader,
            model=model, optimizer=optimizer,
            lr_schedule=lr_schedule, metric=metric,
            logger=logger, logger_all=logger_all,
        )
        logger_all.info(
            f'Rank {dist.get_rank()}/{dist.get_world_size()} '
            f'Train Epoch {epoch} {score}'
        )
        if (
            (dist.get_rank() == 0) and (
                ((epoch + 1) % configs.trainer.save_freq == 0)
                or (epoch == configs.max_epoch - 1)
            )
        ):
            state_dict = model.module.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model': configs.model.kind,
                'score': score,
                'state_dict': state_dict,
            }
            ckpt_path = (
                f'{configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, ckpt_path)
            os.system(f'cp {configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth {configs.ckpt_dir}/latest.pth')
            logger.info(f'Save checkpoint in epoch {epoch}')
        # To avoid deadlock
        dist.barrier()
        time.sleep(2.33)


    if is_test:
        if dist.get_rank() != 0:
            return
        support_dataset = datasets.__dict__['VanillaDataset'](**configs.dataset.meta_kwargs)
        support_loader = torch.utils.data.DataLoader(
            support_dataset, shuffle=False,
            batch_size=configs.dataset.support_batch_size,
            # batch_size=52,
            num_workers=configs.dataset.num_workers,
            pin_memory=False, drop_last=True,
            worker_init_fn=seed_worker
        )
        prepare_prototypes(
            configs=configs, data_loader=support_loader,
            model=model, logger=logger, logger_all=logger_all,
        )
        eval_dataset = datasets.__dict__[configs.dataset.kind](**configs.dataset.eval_kwargs)
        logger.info(eval_dataset.info)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, shuffle=False,
            batch_size=configs.dataset.eval_batch_size,
            num_workers=configs.dataset.num_workers,
            pin_memory=False, drop_last=False
        )
        logger.info(f'{len(eval_dataset)} eval images')
        model.eval()
        score = run_episode(configs=configs, model = model, data_loader = eval_loader,
                metric=metric, logger = logger, logger_all = logger_all)
        logger_all.info(
            f'Rank {dist.get_rank()}/{dist.get_world_size()} '
            f'Eval {score}'
        )
