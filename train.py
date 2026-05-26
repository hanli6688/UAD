import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class UAD_Trainer(nn.Module):
    def __init__(
        self,
        backbone,
        feat_list,
        input_dim,
        output_dim,
        learning_rate,
        device,
        image_size,
        prompting_depth,
        prompting_length,
        prompting_branch,
        prompting_type,
        use_hsf,
        k_clusters,
        use_idag=True,
        idag_intensity=0.5
    ):
        super().__init__()

        self.device = device
        self.lr = learning_rate
        self.use_idag = use_idag
        self.idag_intensity = idag_intensity

        # ===== Backbone (CLIP encoder) =====
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(backbone)
        self.model = self.model.to(device)

        # ===== Prompt Learner =====
        self.prompt_learner = nn.Parameter(
            torch.randn(prompting_length, output_dim)
        )

        # ===== Structural branch =====
        self.struct_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)

        # ===== Optimizer =====
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, images):
        # ===== Extract features =====
        with torch.no_grad():
            image_features = self.model.encode_image(images)

        # ===== Semantic branch =====
        prompt_embed = self.prompt_learner.mean(dim=0)
        semantic_score = torch.matmul(image_features, prompt_embed)

        # ===== Structural branch =====
        B, C = image_features.shape
        struct_feat = image_features.view(B, C, 1, 1)
        struct_feat = self.struct_proj(struct_feat).view(B, -1)

        # ===== Combine =====
        anomaly_score = semantic_score + struct_feat.mean(dim=1)

        return anomaly_score

    # =========================
    # IDAG: anomaly synthesis
    # =========================
    def anomaly_synthesis(self, images):
        noise = torch.randn_like(images) * self.idag_intensity
        synthetic = images + noise
        return torch.clamp(synthetic, 0, 1)

    # =========================
    # Train one epoch
    # =========================
    def train_epoch(self, dataloader):
        self.train()
        total_loss = 0

        for images, _ in tqdm(dataloader):
            images = images.to(self.device)

            # ===== IDAG augmentation =====
            if self.use_idag:
                aug_images = self.anomaly_synthesis(images)
                inputs = torch.cat([images, aug_images], dim=0)
                labels = torch.cat([
                    torch.zeros(images.size(0)),
                    torch.ones(images.size(0))
                ]).to(self.device)
            else:
                inputs = images
                labels = torch.zeros(images.size(0)).to(self.device)

            scores = self.forward(inputs)

            loss = F.binary_cross_entropy_with_logits(
                scores.squeeze(), labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # =========================
    # Evaluation
    # =========================
    @torch.no_grad()
    def evaluation(self, dataloader, cls_names, save_fig, image_dir):
        self.eval()

        results = {}
        all_scores = []

        for images, _ in dataloader:
            images = images.to(self.device)
            scores = self.forward(images)
            all_scores.append(scores.cpu())

        avg_score = torch.cat(all_scores).mean().item()

        results['Average'] = {
            'f1_px': avg_score
        }

        return results

    def save(self, path):
        torch.save(self.state_dict(), path)