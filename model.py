import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================================================
# Utils
# =========================================================

def l2_normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)


def pairwise_cosine(x):
    """
    x: B x L x C
    return: B x L x L
    """
    x = l2_normalize(x, dim=-1)
    return torch.matmul(x, x.transpose(1, 2))


def compute_entropy(p):
    return -torch.sum(p * torch.log(p + 1e-8), dim=-1)


# =========================================================
# Memory Bank
# =========================================================

class MemoryBank(nn.Module):
    def __init__(self, size=100, dim=512, momentum=0.9):
        super().__init__()

        self.size = size
        self.dim = dim
        self.momentum = momentum

        self.register_buffer("bank", torch.randn(size, dim))
        self.bank = l2_normalize(self.bank, dim=1)

        self.ptr = 0

    @torch.no_grad()
    def update(self, features):
        """
        features: N x C
        """
        features = l2_normalize(features, dim=1)

        n = features.shape[0]
        if n >= self.size:
            self.bank = features[:self.size]
            return

        end = self.ptr + n
        if end <= self.size:
            self.bank[self.ptr:end] = features
        else:
            first = self.size - self.ptr
            self.bank[self.ptr:] = features[:first]
            self.bank[:end - self.size] = features[first:]

        self.ptr = end % self.size

    def forward(self):
        return self.bank


# =========================================================
# Structural Modeling (Region-aware)
# =========================================================

class StructuralModel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj = nn.Linear(dim, dim)

    def forward(self, patch_tokens):
        """
        patch_tokens: B x L x C
        return: B x L
        """

        x = self.proj(patch_tokens)

        # local neighborhood similarity
        sim = pairwise_cosine(x)

        # mask diagonal
        eye = torch.eye(sim.shape[-1], device=sim.device)
        sim = sim * (1 - eye)

        # region consistency
        consistency = sim.mean(dim=-1)

        anomaly = 1 - consistency

        return anomaly


# =========================================================
# Patch-level deviation
# =========================================================

class PatchDeviation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, patch_tokens, memory_bank):
        """
        patch_tokens: B x L x C
        memory_bank: M x C
        """

        dist = torch.cdist(patch_tokens, memory_bank.unsqueeze(0))
        min_dist = dist.min(dim=-1)[0]

        return min_dist


# =========================================================
# Semantic Consistency
# =========================================================

class SemanticModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cls_token, text_feature):
        """
        cls_token: B x C
        text_feature: B x 2 x C
        """

        cls_token = l2_normalize(cls_token, dim=-1)
        text_feature = l2_normalize(text_feature, dim=-1)

        sim = torch.matmul(cls_token.unsqueeze(1), text_feature.transpose(1, 2))
        sim = sim.squeeze(1)

        prob = torch.softmax(sim, dim=1)

        anomaly_score = prob[:, 1]

        return anomaly_score, prob


# =========================================================
# Fusion Module
# =========================================================

class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_sem = nn.Parameter(torch.tensor(1.0))
        self.w_struct = nn.Parameter(torch.tensor(1.0))
        self.w_patch = nn.Parameter(torch.tensor(1.0))

    def forward(self, sem, struct, patch):

        struct = struct.mean(dim=1)
        patch = patch.mean(dim=1)

        score = (
            self.w_sem * sem +
            self.w_struct * struct +
            self.w_patch * patch
        )

        return score


# =========================================================
# UAD Main Model
# =========================================================

class UAD(nn.Module):
    def __init__(
        self,
        clip_model,
        feature_dim=512,
        memory_size=100,
        image_size=518
    ):
        super().__init__()

        self.clip_model = clip_model
        self.image_size = image_size

        # modules
        self.memory_bank = MemoryBank(memory_size, feature_dim)
        self.structural = StructuralModel(feature_dim)
        self.patch_dev = PatchDeviation()
        self.semantic = SemanticModel()
        self.fusion = FusionModule()

    # =========================================================
    # Multi-level alignment
    # =========================================================
    def aggregate_patches(self, patch_tokens_list):
        """
        patch_tokens_list: list of [B x L x C]
        """
        aligned = []

        for tokens in patch_tokens_list:
            aligned.append(l2_normalize(tokens, dim=-1))

        return aligned

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, image, class_name):

        cls_token, patch_tokens_list, text_feature = \
            self.clip_model.extract_feat(image, class_name)

        patch_tokens_list = self.aggregate_patches(patch_tokens_list)

        struct_scores = []
        patch_scores = []

        memory = self.memory_bank()

        for tokens in patch_tokens_list:

            struct = self.structural(tokens)
            patch = self.patch_dev(tokens, memory)

            struct_scores.append(struct)
            patch_scores.append(patch)

        struct_scores = torch.stack(struct_scores, dim=1)
        patch_scores = torch.stack(patch_scores, dim=1)

        struct_map = struct_scores.mean(dim=1)
        patch_map = patch_scores.mean(dim=1)

        sem_score, sem_prob = self.semantic(cls_token, text_feature)

        anomaly_map = (struct_map + patch_map) / 2

        B, L = anomaly_map.shape
        H = int(np.sqrt(L))

        anomaly_map = anomaly_map.view(B, 1, H, H)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=self.image_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

        final_score = self.fusion(sem_score, struct_map, patch_map)

        return {
            "anomaly_map": anomaly_map,
            "scores": {
                "semantic": sem_score,
                "structural": struct_map.mean(dim=1),
                "patch": patch_map.mean(dim=1),
                "final": final_score
            },
            "raw": {
                "struct_map": struct_map,
                "patch_map": patch_map,
                "sem_prob": sem_prob
            }
        }

    # =========================================================
    # Memory update (for training)
    # =========================================================
    @torch.no_grad()
    def update_memory(self, patch_tokens_list):

        all_tokens = []

        for tokens in patch_tokens_list:
            B, L, C = tokens.shape
            tokens = tokens.reshape(-1, C)
            all_tokens.append(tokens)

        all_tokens = torch.cat(all_tokens, dim=0)

        self.memory_bank.update(all_tokens)

    # =========================================================
    # Loss (for training)
    # =========================================================
    def compute_loss(self, outputs, gt_label=None, gt_mask=None):

        loss = 0

        sem = outputs["scores"]["semantic"]
        struct = outputs["raw"]["struct_map"]
        patch = outputs["raw"]["patch_map"]

        # semantic loss (binary)
        if gt_label is not None:
            loss_sem = F.binary_cross_entropy(
                sem,
                gt_label.float()
            )
            loss += loss_sem

        # structural regularization
        loss_struct = struct.mean()
        loss += 0.1 * loss_struct

        # patch sparsity
        loss_patch = patch.mean()
        loss += 0.1 * loss_patch

        # pixel-level (if mask exists)
        if gt_mask is not None:
            pred_map = outputs["anomaly_map"]

            pred_map = pred_map / (pred_map.max() + 1e-6)

            loss_pixel = F.binary_cross_entropy(
                pred_map,
                gt_mask.float()
            )

            loss += loss_pixel

        return loss

    # =========================================================
    # Inference helper
    # =========================================================
    def inference(self, image, class_name):

        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, class_name)

        return outputs["anomaly_map"], outputs["scores"]

    # =========================================================
    # Visualization support
    # =========================================================
    def get_visualization_maps(self, outputs):

        return {
            "anomaly": outputs["anomaly_map"],
            "structural": outputs["raw"]["struct_map"],
            "patch": outputs["raw"]["patch_map"]
        }

    # =========================================================
    # Uncertainty Modeling
    # =========================================================

    class UncertaintyEstimator(nn.Module):
        """

        """

        def __init__(self):
            super().__init__()

        def forward(self, score_map):
            """
            score_map: B x L
            """

            prob = torch.sigmoid(score_map)

            entropy = -prob * torch.log(prob + 1e-6) - \
                      (1 - prob) * torch.log(1 - prob + 1e-6)

            return entropy

    # =========================================================
    # Adaptive Fusion（升级版 Eq.14）
    # =========================================================

    class AdaptiveFusion(nn.Module):
        """

        """

        def __init__(self, dim=3):
            super().__init__()

            self.mlp = nn.Sequential(
                nn.Linear(dim, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )

        def forward(self, sem, struct, patch):
            """
            sem: B
            struct: B x L
            patch: B x L
            """

            struct_mean = struct.mean(dim=1)
            patch_mean = patch.mean(dim=1)

            feat = torch.stack([sem, struct_mean, patch_mean], dim=1)

            weights = self.mlp(feat)  # B x 3

            score = (
                    weights[:, 0] * sem +
                    weights[:, 1] * struct_mean +
                    weights[:, 2] * patch_mean
            )

            return score, weights

    # =========================================================
    # Region Refinement
    # =========================================================

    class RegionRefinement(nn.Module):
        """
         anomaly map
        """

        def __init__(self, kernel_size=3):
            super().__init__()

            self.kernel_size = kernel_size

        def forward(self, anomaly_map):
            """
            anomaly_map: B x 1 x H x W
            """

            padding = self.kernel_size // 2

            smooth = F.avg_pool2d(
                anomaly_map,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding
            )

            refined = 0.5 * anomaly_map + 0.5 * smooth

            return refined

    # =========================================================
    # Prompt Ensemble
    # =========================================================

    class PromptEnsemble(nn.Module):
        """

        """

        def __init__(self, clip_model, prompt_templates):
            super().__init__()

            self.clip_model = clip_model
            self.templates = prompt_templates

        def forward(self, class_name):
            text_features = []

            for template in self.templates:
                text = template.format(class_name)

                feat = self.clip_model.encode_text(text)
                feat = F.normalize(feat, dim=-1)

                text_features.append(feat)

            text_features = torch.stack(text_features, dim=0)

            # 平均
            text_feature = text_features.mean(dim=0)

            return text_feature

    # =========================================================
    # Extended UAD Wrapper
    # =========================================================

    class UAD_Enhanced(nn.Module):
        """

        - uncertainty
        - adaptive fusion
        - refinement
        - prompt ensemble
        """

        def __init__(self, base_uad, clip_model):
            super().__init__()

            self.base = base_uad

            self.uncertainty = UncertaintyEstimator()
            self.adaptive_fusion = AdaptiveFusion()
            self.refinement = RegionRefinement()

            self.prompt_ensemble = PromptEnsemble(
                clip_model,
                prompt_templates=[
                    "a photo of a normal {}",
                    "a defective {}",
                    "a damaged {}",
                    "a broken {}"
                ]
            )

        def forward(self, image, class_name):
            outputs = self.base.forward(image, class_name)

            anomaly_map = outputs["anomaly_map"]
            scores = outputs["scores"]

            # =====================================================
            # uncertainty
            # =====================================================
            struct_map = outputs["raw"]["struct_map"]
            uncertainty = self.uncertainty(struct_map)

            # =====================================================
            # adaptive fusion
            # =====================================================
            final_score, weights = self.adaptive_fusion(
                scores["semantic"],
                struct_map,
                outputs["raw"]["patch_map"]
            )

            # =====================================================
            # refinement
            # =====================================================
            anomaly_map = anomaly_map.unsqueeze(1)
            anomaly_map = self.refinement(anomaly_map)
            anomaly_map = anomaly_map.squeeze(1)

            return {
                "anomaly_map": anomaly_map,
                "final_score": final_score,
                "uncertainty": uncertainty.mean(dim=1),
                "fusion_weights": weights
            }