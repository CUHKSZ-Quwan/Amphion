# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, json5
import shutil
import json
import time
import torch
import numpy as np
from utils.util import Logger, ValueWindow
from torch.utils.data import ConcatDataset, DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_trainer import BaseTrainer
from models.base.base_sampler import VariableSampler
from models.tts.gpt_tts.gpt_tts_dataset import (
    GPTTTSDataset,
    GPTTTSCollator,
    batch_by_size,
)
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from models.tts.gpt_tts.gpt_tts import GPTTTS
from models.codec.codec_latent.codec_latent import (
    LatentCodecEncoder,
    LatentCodecDecoderWithTimbre,
)
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from transformers import get_inverse_sqrt_schedule
from tqdm import tqdm
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration


class GPTTTS2Trainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(50)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )
            self.total_train_step = len(self.train_dataloader)

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # setup wav codec encoder, latent codec encoder, latent codec decoder
        self.wav_codec_enc, self.latent_codec_enc, self.latent_codec_dec = (
            self._build_codec()
        )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            (
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.train_dataloader,
                self.valid_dataloader,
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # TODO: Resume from ckpt need test/debug
        with self.accelerator.main_process_first():
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
                end = time.monotonic_ns()
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        model = GPTTTS(cfg=self.cfg.model.gpt_tts)
        return model

    def _build_codec(self):

        wav_codec_enc = CodecEncoder(cfg=self.cfg.model.wav_codec.encoder)
        latent_codec_enc = LatentCodecEncoder(cfg=self.cfg.model.latent_codec.encoder)
        latent_codec_dec = LatentCodecDecoderWithTimbre(
            cfg=self.cfg.model.latent_codec.decoder
        )
        if self.cfg.model.wav_codec.encoder.pretrained_ckpt != '':
            wav_codec_enc.load_state_dict(torch.load(self.cfg.model.wav_codec.encoder.pretrained_ckpt))
            print("Loaded wav_codec_enc: {}".format(self.cfg.model.wav_codec.encoder.pretrained_ckpt))
        if self.cfg.model.latent_codec.encoder.pretrained_ckpt != '':
            latent_codec_enc.load_state_dict(torch.load(self.cfg.model.latent_codec.encoder.pretrained_ckpt))
            print("Loaded latent_codec_enc: {}".format(self.cfg.model.latent_codec.encoder.pretrained_ckpt))
        if self.cfg.model.latent_codec.decoder.pretrained_ckpt != '':
            latent_codec_dec.load_state_dict(torch.load(self.cfg.model.latent_codec.decoder.pretrained_ckpt))
            print("Loaded latent_codec_dec: {}".format(self.cfg.model.latent_codec.decoder.pretrained_ckpt))

        wav_codec_enc.eval()
        latent_codec_enc.eval()
        latent_codec_dec.eval()

        wav_codec_enc.requires_grad_(False)
        latent_codec_enc.requires_grad_(False)
        latent_codec_dec.requires_grad_(False)

        # to device
        wav_codec_enc = wav_codec_enc.to(self.accelerator.device)
        latent_codec_enc = latent_codec_enc.to(self.accelerator.device)
        latent_codec_dec = latent_codec_dec.to(self.accelerator.device)

        return wav_codec_enc, latent_codec_enc, latent_codec_dec

    def _build_dataset(self):
        return GPTTTSDataset, GPTTTSCollator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
            train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.seed(980205)
            np.random.shuffle(batch_sampler)
            print(batch_sampler[:1])
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

            valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            valid_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                valid_dataset.num_frame_indices,
                valid_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]
            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(batches, drop_last=False),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            valid_collate = Collator(self.cfg)

            valid_loader = DataLoader(
                valid_dataset,
                shuffle=True,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.train.lr_warmup_steps,  # TODO: need to check wheather need to multiply by num_processes
        )
        return lr_scheduler

    def _build_criterion(self):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        return criterion

    @staticmethod
    def _count_parameters(model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    # TODO, save without module.
    def save_checkpoint(self, state_dict, saved_model_path):
        torch.save(state_dict, saved_model_path)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint")
        assert os.path.exists(checkpoint_path)
        checkpoint_filename = open(checkpoint_path).readlines()[-1].strip()
        model_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        assert os.path.exists(model_path)
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            self.logger.info(f"Re(store) from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        return checkpoint

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["speech"]
        mask = batch["speech_mask"]
        phone_id = batch["phone_id"]
        phone_id_mask = batch["phone_id_mask"]

        with torch.no_grad():
            vq_emb = self.wav_codec_enc(speech.unsqueeze(1))
            vq_emb = self.latent_codec_enc(vq_emb)
            (
                _,
                vq_indices,
                _,
                _,
                _,
                _,
            ) = self.latent_codec_dec(
                vq_emb, vq=True, eval_vq=False, return_spk_embs=False
            )
            target = vq_indices[0, :, :]
            # release memory
            del speech
            torch.cuda.empty_cache()

        out = self.model(
            phone_ids=phone_id.long(),
            phone_mask=phone_id_mask.long(),
            target_ids=target.long(),
            target_mask=mask.long(),
        )

        # loss
        total_loss += out.loss
        train_losses["ce_loss"] = out.loss

        self.optimizer.zero_grad()
        # total_loss.backward()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return (total_loss.item(), train_losses, train_stats)

    @torch.inference_mode()
    def _valid_step(self, batch):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        speech = batch["speech"]
        mask = batch["speech_mask"]
        phone_id = batch["phone_id"]
        phone_id_mask = batch["phone_id_mask"]

        with torch.no_grad():
            vq_emb = self.wav_codec_enc(speech.unsqueeze(1))
            vq_emb = self.latent_codec_enc(vq_emb)
            (
                _,
                vq_indices,
                _,
                _,
                _,
                _,
            ) = self.latent_codec_dec(
                vq_emb, vq=True, eval_vq=False, return_spk_embs=False
            )
            target = vq_indices[0, :, :]
            # release memory
            del speech
            torch.cuda.empty_cache()

        out = self.model(
            phone_ids=phone_id.long(),
            phone_mask=phone_id_mask.long(),
            target_ids=target.long(),
            target_mask=mask.long(),
        )

        # loss
        total_loss += out.loss
        valid_losses["ce_loss"] = out.loss

        return (total_loss.item(), valid_losses, valid_stats)

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()

        for batch in tqdm(self.valid_dataloader):
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss = total_loss
            for key, value in valid_losses.items():
                epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        for batch in self.train_dataloader:
            # Put the data to cuda device
            start_time = time.time()
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            self.time_window.append(time.time() - start_time)

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (1 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = self._valid_epoch()
            if isinstance(valid_losses, dict):
                for key, loss in valid_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Check if hit save_checkpoint_stride and run_eval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)
                        run_eval |= self.run_eval[i]

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
                print("save state......")
                self.accelerator.save_state(path)
                print("finish saving state......")
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path[idx].append(path)
                    while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                        to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path[idx].insert(0, path)
                    else:
                        do_remove.add(path)

                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    if self.accelerator.is_main_process:
                        self.logger.debug(f"Remove old checkpoint: {path}")

            self.accelerator.wait_for_everyone()
            if run_eval:
                # TODO: run evaluation
                pass

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )
        self.accelerator.end_training()

    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}/{}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.total_train_step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )
        self.logger.info(", ".join(message))
