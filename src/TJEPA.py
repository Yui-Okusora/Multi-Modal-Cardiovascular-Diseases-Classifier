import torch
import torch.nn as nn
from transformers import AutoModel

class VitalsContextEncoder(nn.Module):
    """
    Context Pathway: Compresses the 12-dimensional vitals vector 
    (6 raw vitals + 6 missingness masks) into a latent space.
    Input shape:  [Batch, 12]
    Output shape: [Batch, 128]
    """
    def __init__(self, input_dim=12, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, latent_dim)
        )

    def forward(self, vitals_tensor):
        return self.net(vitals_tensor)


class PhoBERTTargetEncoder(nn.Module):
    """
    Target Pathway: Frozen pre-trained PhoBERT outputting text embeddings.
    Input tokens shape: [Batch, Sequence_Length]
    Output shape:       [Batch, 768]
    """
    def __init__(self, model_name="vinai/phobert-base-v2"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Completely freeze weights to prevent representation collapse
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the <s> token (index 0) vector as the sentence embedding
        text_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [Batch, 768]
        return text_embedding


class PredictorBridge(nn.Module):
    """
    Predictor Bridge: Maps the 128-D vitals space to the 768-D language space.
    Input shape:  [Batch, 128]
    Output shape: [Batch, 768]
    """
    def __init__(self, latent_dim=128, text_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, text_dim)
        )

    def forward(self, context_latent):
        return self.net(context_latent)


class MultimodalDownstreamClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 128 (Vitals Latent) + 768 (PhoBERT Latent) = 896 Dims
        self.classifier = nn.Sequential(
            nn.Linear(128 + 768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, context_latent, text_embedding):
        multimodal_features = torch.cat([context_latent, text_embedding], dim=1)
        return self.classifier(multimodal_features)