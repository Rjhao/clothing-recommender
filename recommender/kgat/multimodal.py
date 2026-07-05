"""Multimodal Feature Module - CLIP + BERT.

This module extracts and fuses visual (CLIP) and textual (BERT) features.

Key formulas:
    f_img = CLIP-ViT(image) -> image embedding
    f_text = BERT(text) -> text embedding
    f_mm = 0.6 * f_img + 0.4 * f_text (fixed fusion in baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from PIL import Image
import numpy as np


class CLIPEncoder(nn.Module):
    """CLIP-based image encoder for visual features.

    Uses CLIP's ViT model to extract image features.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 64,
        freeze: bool = True,
    ):
        """
        Initialize CLIP encoder.

        Args:
            model_name: CLIP model name (or path)
            embedding_dim: Output embedding dimension
            freeze: Whether to freeze CLIP weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Try to import transformers, fallback to mock if not available
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.use_pretrained = True

            if freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            # Projection layer to desired dimension
            self.projection = nn.Linear(
                self.clip_model.config.projection_dim,
                embedding_dim
            )

        except ImportError:
            self.use_pretrained = False
            # Mock CLIP for testing without transformers
            self.projection = nn.Linear(512, embedding_dim)

    def forward(
        self,
        images: Union[torch.Tensor, list],
        return_tensor: bool = True
    ) -> torch.Tensor:
        """
        Extract image features.

        Args:
            images: Image tensor [batch, channels, height, width] or list of PIL Images
            return_tensor: Whether to return tensor or embeddings directly

        Returns:
            Image embeddings [batch, embedding_dim]
        """
        if not self.use_pretrained:
            # Mock CLIP - return random embeddings
            if isinstance(images, torch.Tensor):
                batch_size = images.size(0)
            else:
                batch_size = len(images)
            return torch.randn(batch_size, self.embedding_dim)

        # Process images
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL images
            pil_images = []
            for img in images:
                # Denormalize if needed
                img = img.cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            images = pil_images

        # Process with CLIP
        inputs = self.clip_processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.clip_model.device)

        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs)

        # Project to desired dimension
        embeddings = self.projection(image_embeddings)

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode_text(self, texts: list) -> torch.Tensor:
        """
        Encode text with CLIP.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [batch, embedding_dim]
        """
        if not self.use_pretrained:
            return torch.randn(len(texts), self.embedding_dim)

        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True
        ).to(self.clip_model.device)

        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)

        embeddings = self.projection(text_embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class BERTEncoder(nn.Module):
    """BERT-based text encoder for textual features.

    Extracts text features from product descriptions, reviews, etc.
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        embedding_dim: int = 64,
        freeze: bool = True,
        pooling: str = "cls",
    ):
        """
        Initialize BERT encoder.

        Args:
            model_name: BERT model name
            embedding_dim: Output embedding dimension
            freeze: Whether to freeze BERT weights
            pooling: Pooling strategy ('cls', 'mean', 'max')
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling = pooling

        try:
            from transformers import BertModel, BertTokenizer
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.use_pretrained = True

            if freeze:
                for param in self.bert_model.parameters():
                    param.requires_grad = False

            # Projection layer
            bert_dim = self.bert_model.config.hidden_size
            self.projection = nn.Linear(bert_dim, embedding_dim)

        except ImportError:
            self.use_pretrained = False
            self.projection = nn.Linear(768, embedding_dim)

    def forward(self, texts: Union[list, torch.Tensor]) -> torch.Tensor:
        """
        Extract text features.

        Args:
            texts: List of text strings or tokenized input IDs

        Returns:
            Text embeddings [batch, embedding_dim]
        """
        if not self.use_pretrained:
            if isinstance(texts, torch.Tensor):
                batch_size = texts.size(0)
            else:
                batch_size = len(texts)
            return torch.randn(batch_size, self.embedding_dim)

        # Tokenize if not already
        if isinstance(texts, list):
            inputs = self.bert_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.bert_model.device)
        else:
            inputs = texts

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Pooling
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                # Mask padding tokens
                hidden = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                embeddings = torch.sum(hidden * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling == "max":
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]

        # Project
        embeddings = self.projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class MultimodalFusion(nn.Module):
    """Multimodal feature fusion module.

    Fuses image and text features with learned or fixed weights.

    In KGAT baseline: f_mm = 0.6 * f_img + 0.4 * f_text
    In improvement A (adaptive gate): dynamic weights based on confidence
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        fusion_method: str = "weighted",  # "weighted", "concat", "gate"
    ):
        """
        Initialize multimodal fusion.

        Args:
            embedding_dim: Feature dimension
            fusion_method: Fusion method ('weighted', 'concat', 'gate')
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method

        if fusion_method == "weighted":
            # Fixed weights (baseline KGAT)
            self.register_buffer('image_weight', torch.tensor(0.6))
            self.register_buffer('text_weight', torch.tensor(0.4))

        elif fusion_method == "concat":
            self.projection = nn.Linear(embedding_dim * 2, embedding_dim)

        elif fusion_method == "gate":
            # Gating mechanism (used in improvement A)
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            self.projection = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse multimodal features.

        Args:
            image_features: Image embeddings [batch, dim]
            text_features: Text embeddings [batch, dim]
            confidence: Optional confidence scores [batch] for adaptive fusion

        Returns:
            Fused features [batch, dim]
        """
        if self.fusion_method == "weighted":
            # Fixed weighted fusion (baseline)
            fused = (
                self.image_weight * image_features +
                self.text_weight * text_features
            )

        elif self.fusion_method == "concat":
            # Concatenation + projection
            concat = torch.cat([image_features, text_features], dim=-1)
            fused = self.projection(concat)

        elif self.fusion_method == "gate":
            # Gated fusion with confidence
            concat = torch.cat([image_features, text_features], dim=-1)

            if confidence is not None and self.training:
                # Adaptive: reduce CF weight when confidence is low
                gate = self.gate_network(concat)
                # confidence modulates the gate
                conf_adjust = confidence.unsqueeze(-1).clamp(0.1, 1.0)
                gate = gate * conf_adjust
            else:
                gate = self.gate_network(concat)

            fused = gate * image_features + (1 - gate) * text_features
            fused = self.projection(torch.cat([fused, concat], dim=-1))

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused


class MultimodalModule(nn.Module):
    """Multimodal Feature Module combining CLIP and BERT.

    This is the multimodal branch in KG-HybridRec.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        fusion_method: str = "weighted",
        use_clip: bool = True,
        use_bert: bool = True,
    ):
        """
        Initialize Multimodal Module.

        Args:
            embedding_dim: Embedding dimension
            fusion_method: How to fuse modalities ('weighted', 'concat', 'gate')
            use_clip: Whether to use CLIP for images
            use_bert: Whether to use BERT for text
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_clip = use_clip
        self.use_bert = use_bert

        if use_clip:
            self.clip_encoder = CLIPEncoder(embedding_dim=embedding_dim)
        if use_bert:
            self.bert_encoder = BERTEncoder(embedding_dim=embedding_dim)

        self.fusion = MultimodalFusion(embedding_dim, fusion_method)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[Union[list, torch.Tensor]] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract and fuse multimodal features.

        Args:
            images: Image tensor or list of PIL Images
            texts: List of text strings or tokenized inputs
            confidence: Optional confidence for adaptive fusion

        Returns:
            Tuple of (fused_features, metrics_dict)
        """
        metrics = {}

        # Extract image features
        if images is not None and self.use_clip:
            image_features = self.clip_encoder(images)
            metrics['image_norm'] = image_features.norm(dim=-1).mean().item()
        else:
            # Use zero features if no images
            batch_size = len(texts) if texts is not None else 1
            image_features = torch.zeros(batch_size, self.embedding_dim)

        # Extract text features
        if texts is not None and self.use_bert:
            text_features = self.bert_encoder(texts)
            metrics['text_norm'] = text_features.norm(dim=-1).mean().item()
        else:
            batch_size = len(images) if images is not None else 1
            text_features = torch.zeros(batch_size, self.embedding_dim)

        # Fuse
        fused_features = self.fusion(image_features, text_features, confidence)
        metrics['fused_norm'] = fused_features.norm(dim=-1).mean().item()

        return fused_features, metrics

    def encode_image(self, images) -> torch.Tensor:
        """Encode images only."""
        if not self.use_clip:
            return torch.zeros(len(images), self.embedding_dim)
        return self.clip_encoder(images)

    def encode_text(self, texts) -> torch.Tensor:
        """Encode text only."""
        if not self.use_bert:
            return torch.zeros(len(texts), self.embedding_dim)
        return self.bert_encoder(texts)
