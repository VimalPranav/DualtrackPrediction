import argparse
from src.engine.tracking_estimator import BaseTrackingEstimator
from src.models.model_registry import get_model, register_model

from src.utils.pose import (
    get_absolute_to_global_transforms_torch,
    get_relative_transforms_torch,
    pose_vector_to_rotation_matrix_torch,
    rotation_matrix_to_pose_vector_torch,
)
import torch
from torch import nn
from src.models.bert import BertConfig, BertEncoder


class SpatialMeanPooling(nn.Module):
    def forward(self, x):
        return x.mean((-1, -2))


class SimpleModelForSparseTrackingEstimation(BaseTrackingEstimator, nn.Module):
    """
    Simple model for sparse tracking estimation. This model takes in a sequence of images and outputs a sequence of
    6-DoF poses. The model is based on a backbone that extracts features from the input images. The features are then
    processed by a transformer model to predict the 6-DoF poses.

    Args:
        backbone (nn.Module): Backbone model that extracts features from the input images.
        n_features (int): Number of features extracted by the backbone.
        feature_map_size (int): Size of the feature maps extracted by the backbone.
        patch_size (int): Patch size used by the ViT model (only used if spatial_pool_mode is set to "vit").
        max_seq_length (int): Maximum sequence length used by the transformer model.
        hidden_size (int): Hidden size used by the transformer model.
        pred_mode (str): Prediction mode. Can be one of "local", "global", or "absolute".
            "local" predicts the relative pose between consecutive frames, "global" predicts the global pose (i.e., relative to the first frame) of each frame, and "absolute" predicts the absolute pose of each frame.
        num_hidden_layers (int): Number of hidden layers used by the transformer model.
        spatial_pool_mode (str): Spatial pool mode. Can be one of "vit" or "mean".
        **kwargs: Additional keyword arguments passed to the transformer model.
    """

    def __init__(
        self,
        backbone,
        n_features=512,
        max_seq_length=1024,
        hidden_size=512,
        pred_mode="local",
        num_hidden_layers=4,
        position_embedding_type="absolute",
        features_only=False,
        **kwargs,
    ):
        super().__init__()
        self.features_only = features_only
        self.backbone = backbone
        self.pos_emb = (
            nn.Embedding(max_seq_length+1, hidden_size)
            if position_embedding_type == "absolute"
            else None
        )
        self.pred_mode = pred_mode
        self.position_embedding_type = position_embedding_type
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.proj = torch.nn.Linear(n_features, hidden_size)
        cfg = BertConfig(
            attn_implementation="eager",
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            **kwargs,
        )
        self.bert = BertEncoder(cfg)
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, images, sample_indices):
        feats = self.backbone(images)  # B N C
        feats = self.proj(feats)

        B = feats.shape[0]
        cls = self.cls_token.expand(B, -1, -1)

        feats = torch.cat([cls, feats], dim=1)

        if self.pos_emb is not None:
            cls_indices = torch.zeros(
                B, 1,
                dtype=sample_indices.dtype,
                device=sample_indices.device
            )

            frame_indices = sample_indices + 1

            position_indices = torch.cat(
                [cls_indices, frame_indices],
                dim=1
            )

            pos_emb = self.pos_emb(position_indices)
            feats = feats + pos_emb

        bert_outputs = self.bert(
            feats
        ).last_hidden_state

        cls_feature = bert_outputs[:, 0, :]
        frame_features = bert_outputs[:, 1:, :]

        if self.features_only:
            return {
                "frame_features": frame_features,
                "cls_feature": cls_feature,
            }

        if self.pred_mode == "local":
            return self.fc(frame_features)

        elif self.pred_mode == "global":

            pred = self.fc(frame_features)

            out = {
                "global": pred
            }
            pred_matrix = pose_vector_to_rotation_matrix_torch(pred)
            pred_local = get_relative_transforms_torch(pred_matrix)
            pred_local = rotation_matrix_to_pose_vector_torch(pred_local)

            out["local"] = pred_local.detach()

            return out

        elif self.pred_mode == "absolute":

            pred = self.fc(frame_features)
            out = {
                "absolute": pred
            }
            pred_detached = pred.detach()
            pred_matrix = pose_vector_to_rotation_matrix_torch(
                pred_detached
            )
            
            pred_glob = get_absolute_to_global_transforms_torch(
                pred_matrix
            )[..., 1:, :, :]
            pred_glob = rotation_matrix_to_pose_vector_torch(
                pred_glob
            )

            out["global"] = pred_glob

            pred_loc = get_relative_transforms_torch(
                pred_matrix
            )
            pred_loc = rotation_matrix_to_pose_vector_torch(
                pred_loc
            )

            out["local"] = pred_loc
            return out

        else:
            raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def predict(self, batch):
        device = next(self.parameters()).device
        return self(batch["images"].to(device), batch["sample_indices"].to(device))

    def get_loss(self, batch, pred=None): 
        device = next(self.parameters()).device

        pred = pred if pred is not None else self.predict(batch)

        targets = batch["targets"].to(device)
        padding_lengths = batch["padding_size"]
        crit = nn.MSELoss(reduction="none")

        def _get_loss(pred, targets):
            B, N, D = (
                pred.shape
                if isinstance(pred, torch.Tensor)
                else list(pred.values())[0].shape
            )

            mask = torch.ones(B, N, D, dtype=torch.bool, device=device)
            for i, padding_length in enumerate(padding_lengths):
                if padding_length > 0:
                    mask[i, -padding_length:, :] = 0

            loss = crit(pred, targets)
            masked_loss = torch.where(mask, loss, torch.nan)
            mse_loss_val = masked_loss.nanmean()
            return mse_loss_val

        if isinstance(pred, dict):
            loss = torch.tensor(0.0, device=device)
            if "local" in pred:
                loss += _get_loss(pred["local"], targets)
            if "global" in pred:
                loss += _get_loss(pred["global"], batch["targets_global"].to(device))
            if "absolute" in pred:
                loss += _get_loss(
                    pred["absolute"], batch["targets_absolute"].to(device)
                )
            return loss
        else:
            return _get_loss(pred, targets)



@register_model
def simple_rn_no_temporal_backbone():
    from src.models.video_resnet import (
        VideoResnetWrapperForFeatureMaps,
        video_rn18_no_temporal,
    )

    return VideoResnetWrapperForFeatureMaps(video_rn18_no_temporal())



@register_model
def global_encoder_cnn(
    backbone_weights=None, image_size=224, feature_map_size=14, **kwargs
):

    backbone = nn.Sequential(
        simple_rn_no_temporal_backbone(), 
        SpatialMeanPooling()
    )

    return SimpleModelForSparseTrackingEstimation(
        backbone,
        n_features=512,
        feature_map_size=feature_map_size,
        hidden_size=512,
        features_only=True,
        **kwargs,
    )