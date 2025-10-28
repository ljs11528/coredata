import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import random
from functools import partial
from copy import deepcopy

from models import (build_model, build_ema_model, build_optimizer, get_accuracy, interleave, get_ramp_up, load_pretrain,
                    build_lr_scheduler, get_trainable_params, get_mixmatch_function)
from loss import (build_supervised_loss, build_unsupervised_loss, build_pair_loss)
from models.utils import unwrap_model, consume_prefix_in_state_dict_if_present
from utils import get_device
from loss.visualization import get_pair_info

# for type hint
from typing import Tuple, Optional, Union, Dict, Any, Set
from argparse import Namespace
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from loss.types import LossInfoType
from models.types import LRSchedulerType, OptimizerParametersType


class SimPLEEstimator:
    def __init__(self,
                 exp_args: Namespace,
                 augmenter: Module,
                 strong_augmenter: Module,
                 val_augmenter: Module,
                 num_classes: int,
                 in_channels: int,
                 device: Optional[torch.device] = None):
        self.exp_args = exp_args

        # augmenter
        self.augmenter = augmenter
        self.strong_augmenter = strong_augmenter
        self.val_augmenter = val_augmenter

        self.ema_decay: float = self.exp_args.ema_decay

        self.use_ema: bool = self.exp_args.use_ema
        self.ema_type: str = self.exp_args.ema_type

        if device is None:
            device = get_device(self.exp_args.device)
        self._device = device

        self.model = build_model(
            model_type=self.exp_args.model_type,
            in_channels=in_channels,
            out_channels=num_classes)

        if self.exp_args.use_pretrain:
            load_pretrain(model=self.get_model(),
                          checkpoint_path=self.exp_args.checkpoint_path,
                          allowed_prefix="",
                          ignored_prefix="fc",
                          device=torch.device("cpu"))

        if self.use_ema:
            self.model = build_ema_model(model=self.model, ema_type=self.ema_type, ema_decay=self.ema_decay)

        self.optimizer = self.build_optimizer(
            params=self.get_trainable_params(
                classifier_prefix={"model.fc", "shadow.fc", "fc"}))
        self.lr_scheduler = self.build_lr_scheduler(optimizer=self.optimizer)

        # loss function
        self.ramp_up = get_ramp_up(ramp_up_type=self.exp_args.ramp_up_type,
                                   length=self.max_warmup_step)

        self.lambda_u: float = self.exp_args.lambda_u
        self.lambda_pair: float = self.exp_args.lambda_pair

        # train loss
        self.supervised_loss = build_supervised_loss(self.exp_args)
        self.unsupervised_loss = build_unsupervised_loss(self.exp_args)
        self.pair_loss = build_pair_loss(self.exp_args)

        # val loss
        self.val_loss_fn = nn.CrossEntropyLoss()

        # mixmatch
        self.mixmatch_fn = get_mixmatch_function(
            args=self.exp_args,
            num_classes=num_classes,
            augmenter=self.augmenter,
            strong_augmenter=self.strong_augmenter)

        # visualization function
        self.get_pair_info = partial(get_pair_info,
                                     similarity_metric=self.pair_loss.get_similarity,
                                     confidence_threshold=self.pair_loss.confidence_threshold,
                                     similarity_threshold=self.pair_loss.similarity_threshold)

        # stats
        self._global_step: int = 0

        # move to device
        self.to(self.device)

        self.use_csl: bool = getattr(self.exp_args, "use_csl", True)   # CSL: Class-Specific Loss


    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        if device != self.device:
            self._device = device

            self.to(self.device)

    @property
    def num_epochs(self) -> int:
        return self.exp_args.num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int) -> None:
        assert num_epochs >= 1

        # update configs
        self.exp_args.num_epochs = num_epochs

    @property
    def num_warmup_epochs(self) -> int:
        return self.exp_args.num_warmup_epochs

    @num_warmup_epochs.setter
    def num_warmup_epochs(self, num_warmup_epochs: int) -> None:
        assert num_warmup_epochs >= 0

        # update configs
        self.exp_args.num_warmup_epochs = num_warmup_epochs

        # update ramp-up info
        self.ramp_up.length = self.max_warmup_step

    @property
    def num_step_per_epoch(self) -> int:
        return self.exp_args.num_step_per_epoch

    @num_step_per_epoch.setter
    def num_step_per_epoch(self, num_step_per_epoch: int) -> None:
        assert num_step_per_epoch >= 1

        # update configs
        self.exp_args.num_step_per_epoch = num_step_per_epoch

        # update ramp-up info
        self.ramp_up.length = self.max_warmup_step

    @property
    def log_interval(self) -> int:
        # restrict log_interval to be <= num_step_per_epoch
        return min(self.exp_args.log_interval, self.num_step_per_epoch)

    @property
    def max_grad_norm(self) -> Optional[float]:
        return self.exp_args.max_grad_norm

    @property
    def global_step(self) -> int:
        return self._global_step

    @global_step.setter
    def global_step(self, global_step: int) -> None:
        assert 0 <= global_step <= self.max_step, f"expecting 0 <= global_step" \
                                                  f" <= {self.max_step} but get {global_step}"
        self._global_step = global_step

        # update ramp-up info
        self.ramp_up.current = self.global_step

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_step_per_epoch

    @property
    def max_step(self) -> int:
        return self.num_epochs * self.num_step_per_epoch

    @property
    def max_warmup_step(self) -> int:
        return self.num_warmup_epochs * self.num_step_per_epoch

    @property
    def return_plot_info(self) -> bool:
        return (self.global_step + 1) % self.log_interval == 0 or (self.global_step + 1) % self.num_step_per_epoch == 0

    def to(self, device: torch.device) -> 'SimPLEEstimator':
        self.model.to(device)
        self.augmenter.to(device)
        self.strong_augmenter.to(device)
        self.val_augmenter.to(device)

        return self

    def get_model(self) -> Module:
        return unwrap_model(self.model)

    def get_trainable_model(self) -> Module:
        if self.use_ema:
            return unwrap_model(self.get_model().model)
        else:
            return self.get_model()

    def get_trainable_params(self, classifier_prefix: Union[str, Set[str]]) -> OptimizerParametersType:
        return get_trainable_params(
            model=self.model,
            learning_rate=self.exp_args.learning_rate,
            feature_learning_rate=self.exp_args.feature_learning_rate,
            classifier_prefix=classifier_prefix,
            requires_grad_only=True)

    def build_optimizer(self, params: OptimizerParametersType) -> Optimizer:
        return build_optimizer(
            optimizer_type=self.exp_args.optimizer_type,
            params=params,
            learning_rate=self.exp_args.learning_rate,
            weight_decay=self.exp_args.weight_decay,
            momentum=self.exp_args.optimizer_momentum)

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRSchedulerType:
        return build_lr_scheduler(
            scheduler_type=self.exp_args.lr_scheduler_type,
            optimizer=optimizer,
            max_iter=self.max_step,
            cosine_factor=self.exp_args.lr_cosine_factor,
            step_size=self.exp_args.lr_step_size,
            gamma=self.exp_args.lr_gamma,
            num_warmup_steps=self.exp_args.lr_warmup_step)

    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Tuple[Tensor, LossInfoType]:
        outputs = self.preprocess_batch(batch, batch_idx)

        model_outputs = self.compute_train_logits(x_inputs=outputs["x_inputs"], u_inputs=outputs["u_inputs"])

        outputs.update(model_outputs)

        # calculate loss
        loss, log_dict = self.compute_train_loss(
            x_logits=outputs["x_logits"],
            x_targets=outputs["x_targets"],
            u_logits=outputs["u_logits"],
            u_targets=outputs["u_targets"],
            u_true_targets=outputs["u_true_targets"])

        # save additional logging info and plots
        extra_log_info = self.visualize_loss(
            u_targets=outputs["u_targets"],
            u_true_targets=outputs["u_true_targets"]
        )

        log_dict["log"].update(extra_log_info["log"])
        log_dict["plot"].update(extra_log_info["plot"])

        return loss, log_dict

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        # unpack batch
        x, y = batch

        # move to device
        x = x.to(self.device)
        y = y.to(self.device)

        x = self.val_augmenter(x)

        x_out: Tensor = self.model(x)
        loss = self.val_loss_fn(x_out, y)
        top1_acc, top5_acc = get_accuracy(x_out, y, top_k=(1, 5))

        return dict(
            loss=loss,
            top1_acc=top1_acc,
            top5_acc=top5_acc,
        )

    def preprocess_batch(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Dict[str, Tensor]:
        # unpack batch
        (x_inputs, x_targets), (u_inputs, u_true_targets) = batch

        batch_size = len(x_inputs)

        # load data to device
        x_inputs = x_inputs.to(self.device)
        x_targets = x_targets.to(self.device)
        u_inputs = u_inputs.to(self.device)
        u_true_targets = u_true_targets.to(self.device)

        outputs = self.mixmatch_fn(
            model=self.model,
            **self.mixmatch_fn.preprocess(
                x_inputs=x_inputs,
                x_strong_inputs=x_inputs,
                x_targets=x_targets,
                u_inputs=u_inputs,
                u_strong_inputs=u_inputs,
                u_true_targets=u_true_targets))

        assert len(outputs["x_mixed"]) == len(outputs["p_mixed"]) == batch_size
        assert len(outputs["u_mixed"]) == len(outputs["q_mixed"])

        return dict(
            x_inputs=outputs["x_mixed"],
            x_targets=outputs["p_mixed"],
            u_inputs=outputs["u_mixed"],
            u_targets=outputs["q_mixed"],
            u_true_targets=outputs["q_true_mixed"],
        )

    def compute_train_logits(self, x_inputs: Tensor, u_inputs: Tensor) -> Dict[str, Tensor]:
        batch_size = len(x_inputs)

        # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
        batch_outputs = [x_inputs, *torch.split(u_inputs, batch_size, dim=0)]
        batch_outputs = interleave(batch_outputs, batch_size)

        batch_outputs = [self.model(batch_output) for batch_output in batch_outputs]

        # put interleaved samples back
        batch_outputs = interleave(batch_outputs, batch_size)
        x_logits = batch_outputs[0]
        u_logits = torch.cat(batch_outputs[1:], dim=0)

        return dict(
            x_logits=x_logits,
            u_logits=u_logits,
        )

    def compute_train_loss(self,
                           x_logits: Tensor,
                           x_targets: Tensor,
                           u_logits: Tensor,
                           u_targets: Tensor,
                           u_true_targets: Tensor) -> Tuple[Tensor, LossInfoType]:
        """

        Args:
            x_logits: (labeled batch size, num classes) model output of the labeled data
            x_targets: (labeled batch size, num classes) labels distribution for labeled data
            u_logits: (unlabeled batch size, num classes) model output for unlabeled data
            u_targets: (unlabeled batch size, num classes) guessed labels distribution for unlabeled data
            u_true_targets: (unlabeled batch size, num classes) ground truth labels distribution for unlabeled data,
                this is only used for visualization

        Returns:

        """
        x_probs = F.softmax(x_logits, dim=1)
        u_probs = F.softmax(u_logits, dim=1)

        loss_x = self.supervised_loss(x_logits, x_probs, x_targets)

        # init log info dict
        log_info = dict(loss_x=loss_x.detach().clone())
        plot_info = dict()

        # get current ramp-up value
        ramp_up_value = self.ramp_up(current=self.global_step)

        loss = loss_x

        if self.lambda_u != 0:
            weighted_loss_u, loss_u_log = self.compute_unsupervised_loss(logits=u_logits,
                                                                         probs=u_probs,
                                                                         targets=u_targets,
                                                                         ramp_up_value=ramp_up_value)
            loss += weighted_loss_u

            log_info.update(loss_u_log)

        if self.lambda_pair != 0:
            weighted_loss_pair, loss_pair_log = self.compute_pair_loss(logits=u_logits,
                                                                       probs=u_probs,
                                                                       targets=u_targets,
                                                                       ramp_up_value=ramp_up_value)
            loss += weighted_loss_pair

            log_info.update(loss_pair_log)

        # save final loss value
        log_info["loss"] = loss.detach().clone()

        return loss, dict(
            log=log_info,
            plot=plot_info,
        )

    # def compute_unsupervised_loss(self, logits: Tensor, probs: Tensor, targets: Tensor, ramp_up_value: float) \
    #         -> Tuple[Tensor, Dict[str, Tensor]]:
    #     loss = self.unsupervised_loss(logits, probs, targets)
    #     weighted_loss = ramp_up_value * self.lambda_u * loss

    #     return weighted_loss, dict(
    #         loss_u=loss.detach().clone(),
    #         weighted_loss_u=weighted_loss.detach().clone(),
    #     )
    
    def compute_unsupervised_loss(self, logits: Tensor, probs: Tensor, targets: Tensor, ramp_up_value: float) \
        -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.use_csl:
            per_sample_loss = self.unsupervised_loss(logits, probs, targets, reduction='none')  # shape [batch]
            weight = self.compute_csl_weight(probs, num_classes=self.exp_args.num_classes)
            weighted_loss = (per_sample_loss * weight).mean()
        else:
            weighted_loss = self.unsupervised_loss(logits, probs, targets)

        final_loss = ramp_up_value * self.lambda_u * weighted_loss

        return final_loss, dict(
            loss_u=weighted_loss.detach().clone(),
            weighted_loss_u=final_loss.detach().clone(),
        )

    @torch.no_grad()
    def compute_csl_weight(self, probs: Tensor, num_classes: int, epsilon=1e-8, alpha=2.0) -> Tensor:
        """
        Compute CSL-based weights for image classification task.
        Args:
            probs: Tensor, shape [batch, num_classes], softmax probabilities
        Returns:
            weight: Tensor, shape [batch]
        """
        batch_size = probs.size(0)

        # Step 1: max confidence & corresponding index
        max_confidence, max_indices = probs.max(dim=1)  # [batch]

        # Step 2: compute g_j
        g_j = (num_classes - 1) ** 2 / (2 * (1 - max_confidence + epsilon))  # [batch]

        # Step 3: compute residual variance (exclude max class)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, max_indices.unsqueeze(1), False)
        remaining_preds = probs.masked_fill(~mask, float('nan'))
        mean_remaining = torch.nanmean(remaining_preds, dim=1)  # [batch]
        residual_variance = torch.nanmean((remaining_preds - mean_remaining.unsqueeze(1)) ** 2, dim=1)  # [batch]

        # Step 4: scaled residual variance
        scaled_residual_variance = g_j * residual_variance  # [batch]

        # Step 5: batch_class_stats with SVD clustering
        features = torch.stack([max_confidence, scaled_residual_variance], dim=-1)  # [batch, 2]
        means, vars = self._batch_class_stats(features, num_clusters=2)

        # Pick the cluster with highest confidence mean
        conf_mean, res_mean = means[0]
        conf_var, res_var = vars[0]

        # Step 6: Z-score
        conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
        res_z = (res_mean - scaled_residual_variance) / torch.sqrt(res_var + epsilon)

        # Step 7: weight = exp(-z^2/alpha)
        weight_conf = torch.exp(- (conf_z ** 2) / alpha)
        weight_res = torch.exp(- (res_z ** 2) / alpha)
        weight = weight_conf * weight_res

        # Step 8: confident mask (positive z-score -> weight=1)
        confident_mask = (conf_z > 0) | (res_z > 0)
        weight = torch.where(confident_mask, torch.ones_like(weight), weight)

        return weight
    
    @torch.no_grad()
    def _batch_class_stats(self, features: Tensor, num_clusters: int = 2):
        """
        SVD-based clustering for [batch, 2] features.
        Returns:
            means: [num_clusters, 2]
            vars: [num_clusters, 2]
        """
        eigenvectors = self._compute_eigenvectors_with_svd(features, num_clusters)
        class_assignments = torch.argmax(torch.abs(eigenvectors), dim=1)

        means = []
        vars = []
        for class_id in range(num_clusters):
            points = features[class_assignments == class_id]
            if points.size(0) == 0:
                means.append(torch.zeros(2, device=features.device))
                vars.append(torch.ones(2, device=features.device))
            else:
                means.append(points.mean(dim=0))
                vars.append(points.var(dim=0, unbiased=True))
        means = torch.stack(means)
        vars = torch.stack(vars)

        # Pick cluster with max conf mean
        max_conf_idx = torch.argmax(means[:, 0])
        
        return means[max_conf_idx:max_conf_idx+1], vars[max_conf_idx:max_conf_idx+1]
    
    @torch.no_grad()
    def _compute_eigenvectors_with_svd(self, X: Tensor, num_clusters: int):
        # X: [batch, 2]
        U, S, Vt = torch.linalg.svd(X.T, full_matrices=False)
        eigvals = S ** 2
        idx = torch.argsort(-eigvals)
        eigvecs = Vt.T[:, idx[:num_clusters]]
        return eigvecs


    def compute_pair_loss(self,
                          logits: Tensor,
                          probs: Tensor,
                          targets: Tensor,
                          ramp_up_value: float) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss = self.pair_loss(logits=logits,
                              probs=probs,
                              targets=targets)
        weighted_loss = ramp_up_value * self.lambda_pair * loss

        return weighted_loss, dict(
            loss_pair=loss.detach().clone(),
            weighted_loss_pair=weighted_loss.detach().clone(),
        )

    def visualize_loss(self, u_targets: Tensor, u_true_targets: Tensor) -> LossInfoType:
        return self.get_pair_info(targets=u_targets,
                                  true_targets=u_true_targets,
                                  return_plot_info=self.return_plot_info)

    def training_epoch_end(self, *args, **kwargs) -> None:
        pass

    def get_checkpoint(self) -> Dict[str, Any]:
        checkpoint = dict(
            args=self.exp_args,
            network_state=self.get_model().state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            lr_scheduler_state=self.lr_scheduler.state_dict(),
            ramp_state=self.ramp_up.state_dict(),
            global_step=self.global_step,
            # save random state
            torch_rng_state=torch.get_rng_state(),
            numpy_random_state=np.random.get_state(),
            python_random_state=random.getstate(),
        )

        return checkpoint

    def load_checkpoint(self, checkpoint: Dict[str, Any], recover_optimizer: bool = True,
                        recover_train_progress: bool = True) -> 'SimPLEEstimator':
        # remove DP/DDP wrapper
        network_state = deepcopy(checkpoint["network_state"])
        consume_prefix_in_state_dict_if_present(network_state, prefix="module.")

        self.get_model().load_state_dict(network_state)

        if recover_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            if "lr_scheduler_state" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        if recover_train_progress:
            self.global_step = checkpoint["global_step"]

            if "ramp_state" in checkpoint:
                self.ramp_up.load_state_dict(checkpoint["ramp_state"])
            else:
                self.ramp_up.current = self.global_step
                self.ramp_up.length = self.max_warmup_step

        return self

    @classmethod
    def from_checkpoint(cls,
                        augmenter: Module,
                        strong_augmenter: Module,
                        val_augmenter: Module,
                        checkpoint_path: str,
                        num_classes: int,
                        in_channels: int,
                        device: Optional[torch.device] = None,
                        args_override: Optional[Namespace] = None,
                        recover_train_progress: bool = True,
                        recover_random_state: bool = True) -> 'SimPLEEstimator':
        """

        Args:
            augmenter: (weak) augmenter
            strong_augmenter: strong augmenter
            val_augmenter: augmenter for validation/testing
            checkpoint_path: path to checkpoint
            num_classes: number of classes
            in_channels: number of input channel
            device: if None, will use device in the checkpoint; else will use this device
            args_override: if not None, override the recovered args
            recover_train_progress: if True, will recover global_step and ramp-up state
            recover_random_state: if True, will recover random state

        Returns:

        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        recovered_args = checkpoint["args"]
        if args_override is None:
            args = recovered_args
        else:
            # override args
            args = args_override

        estimator = cls(exp_args=args,
                        augmenter=augmenter,
                        strong_augmenter=strong_augmenter,
                        val_augmenter=val_augmenter,
                        num_classes=num_classes,
                        in_channels=in_channels,
                        device=device)
        estimator.load_checkpoint(checkpoint, recover_optimizer=True, recover_train_progress=recover_train_progress)

        if recover_random_state:
            # recover random state
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])

            if "python_random_state" in checkpoint:
                random.setstate(checkpoint["python_random_state"])

        print(f"Estimator recovered from \"{checkpoint_path}\"", flush=True)

        return estimator
