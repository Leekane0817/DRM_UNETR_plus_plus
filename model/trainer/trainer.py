"""
ACDC Trainer — UNETR++ with Spectral-Dynamic Routing Mixer (SDRM) on ACDC dataset

ACDC dataset: 4 classes (BG, RV, MYO, LV), 100 patients
Uses 70/10/20 train/val/test split as in the paper.

Based on unetr_pp.training.network_training.Trainer_acdc but uses SDRM model.
"""

import math
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from unetr_pp.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from unetr_pp.training.loss_functions.deep_supervision import MultipleOutputLoss2
from unetr_pp.utilities.to_torch import maybe_to_torch, to_cuda
from unetr_pp.network_architecture.initialization import InitWeights_He
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params, get_patch_size, default_3D_augmentation_params
)
from unetr_pp.training.dataloading.dataset_loading import unpack_dataset
from unetr_pp.training.network_training.Trainer_acdc import Trainer_acdc
from unetr_pp.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

# SDRM model
from model.unetr_pp_sdrm import UNETR_PP_SDRM


class unetr_pp_trainer_acdc_sdrm(Trainer_acdc):
    """Trainer for UNETR_PP_SDRM on ACDC cardiac dataset."""

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 2e-4
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False
        self.warmup_epochs = 50

        self.load_plans_file()

        if len(self.plans['plans_per_stage']) == 2:
            Stage = 1
        else:
            Stage = 0

        self.crop_size = [16, 160, 160]  # ACDC patch size
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1  # +1 for background
        self.conv_op = nn.Conv3d
        self.deep_supervision = True

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['plans_per_stage'][0]['patch_size'] = np.array([16, 160, 160])
            self.crop_size = np.array([16, 160, 160])
            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'] = [
                [1, 4, 4], [2, 2, 2], [2, 2, 2]
            ]
            self.process_plans(self.plans)
            self.setup_DA_params()

            if self.deep_supervision:
                net_numpool = len(self.net_num_pool_op_kernel_sizes)
                weights = np.array([1.0, 0.7, 0.4][:net_numpool])
                weights = weights / weights.sum()
                print(f"DS loss weights (SDRM ACDC): {weights}")
                self.ds_loss_weights = weights
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans['data_identifier'] + "_stage%d" % self.stage
            )
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train, seeds_val=seeds_val,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            # torch.compile wrapper check
            _net = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
            assert isinstance(_net, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        self.network = UNETR_PP_SDRM(
            in_channels=self.input_channels,
            out_channels=self.num_classes,       # 4 (BG + RV + MYO + LV)
            img_size=self.crop_size,             # [16, 160, 160]
            feature_size=16,
            num_heads=4,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            do_ds=True,
            stem_kernel_size=(1, 4, 4),          # ACDC: z-axis not downsampled in stem
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"[SDRM ACDC] Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.initial_lr,
            weight_decay=1e-4, betas=(0.9, 0.999),
        )
        self.lr_scheduler = None

    # ------------------------------------------------------------------
    # Training & Validation
    # ------------------------------------------------------------------

    def run_online_evaluation(self, output, target):
        if self.deep_supervision:
            target = target[0]; output = output[0]
        else:
            target = target; output = output
        return super().run_online_evaluation(output, target)

    def run_training(self):
        self.maybe_update_lr(self.epoch)
        ds = self.network.do_ds
        self.network.do_ds = True if self.deep_supervision else False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']; target = data_dict['target']
        data = maybe_to_torch(data); target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data); target = to_cuda(target)
        self.optimizer.zero_grad()
        if self.fp16:
            with autocast():
                output = self.network(data); del data
                l = self.loss(output, target)
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data); del data
            l = self.loss(output, target)
            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target
        return l.detach().cpu().numpy()

    def validate(self, do_mirroring=True, use_sliding_window=True,
                 step_size=0.5, save_softmax=True, use_gaussian=True,
                 overwrite=True, validation_folder_name='validation_raw',
                 debug=False, all_in_gpu=False, segmentation_export_kwargs=None,
                 run_postprocessing_on_folds=True):
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                               step_size=step_size, save_softmax=save_softmax,
                               use_gaussian=use_gaussian, overwrite=overwrite,
                               validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu,
                               segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(
        self, data, do_mirroring=True, mirror_axes=None,
        use_sliding_window=True, step_size=0.5, use_gaussian=True,
        pad_border_mode='constant', pad_kwargs=None, all_in_gpu=False,
        verbose=True, mixed_precision=True,
    ):
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window, step_size=step_size,
            use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
            verbose=verbose, mixed_precision=mixed_precision,
        )
        self.network.do_ds = ds
        return ret

    # ------------------------------------------------------------------
    # Data Split & Augmentation
    # ------------------------------------------------------------------

    def do_split(self):
        """
        ACDC 70/10/20 split as stated in the paper [5].
        Test:  20 patients (excluded from dataset)
        Val:   10 patients
        Train: 70 patients
        """
        # 20 test patients (from imagesTs, excluded)
        TEST_PATIENTS = [
            'patient002','patient003','patient008','patient009','patient012',
            'patient014','patient017','patient024','patient042','patient048',
            'patient049','patient053','patient055','patient064','patient067',
            'patient079','patient081','patient088','patient092','patient095'
        ]
        # 10 validation patients
        VAL_PATIENTS = [
            'patient001','patient004','patient005','patient006','patient007',
            'patient010','patient011','patient013','patient015','patient016'
        ]
        val_keys = []; tr_keys = []
        for k in sorted(self.dataset.keys()):
            pid = k.split('_frame')[0]
            if pid in TEST_PATIENTS:
                continue  # exclude test patients entirely
            elif pid in VAL_PATIENTS:
                val_keys.append(k)
            else:
                tr_keys.append(k)
        tr_keys.sort(); val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys: self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys: self.dataset_val[i] = self.dataset[i]
        self.print_to_log_file(f"70/10/20 split: train={len(tr_keys)}, val={len(val_keys)}, test=40 frames (excluded)")

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:], self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'], self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size, self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'], self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size
        self.data_aug_params["scale_range"] = (0.85, 1.15)
        self.data_aug_params["do_elastic"] = True
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform
        self.data_aug_params["num_cached_per_thread"] = 2

    # ------------------------------------------------------------------
    # Learning Rate Schedule (warmup + cosine decay)
    # ------------------------------------------------------------------

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        if ep < self.warmup_epochs:
            lr = self.initial_lr * (ep / self.warmup_epochs)
        else:
            progress = (ep - self.warmup_epochs) / (self.max_num_epochs - self.warmup_epochs)
            lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        self.optimizer.param_groups[0]['lr'] = lr
        self.print_to_log_file("lr:", np.round(lr, decimals=8))

    def on_epoch_end(self):
        ret = super().on_epoch_end()
        if ret and self.epoch % 10 == 0:
            checkpoint = {
                'epoch': self.epoch,
                'state_dict': self.network.state_dict(),
                'best_val_eval_criterion_MA': self.best_val_eval_criterion_MA,
            }
            torch.save(checkpoint, join(self.output_folder, "model_latest.model"))
            self.print_to_log_file(f"Latest checkpoint saved at epoch {self.epoch}")
        return ret
