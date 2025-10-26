"""
Adaptive Multi-Modal AI Framework for Self-Regulated Learning
Main framework implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import yaml


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder for processing different data modalities"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Text encoder (BERT-based)
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(
            config['model']['text_encoder']['model_name']
        )
        self.text_projection = nn.Linear(
            config['model']['text_encoder']['hidden_size'],
            config['model']['common_dim']
        )
        
        # Visual encoder (ResNet + LSTM)
        self.visual_cnn = self._build_visual_cnn(config)
        self.visual_lstm = nn.LSTM(
            input_size=config['model']['visual_encoder']['cnn_output_dim'],
            hidden_size=config['model']['visual_encoder']['lstm_hidden'],
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.visual_projection = nn.Linear(
            config['model']['visual_encoder']['lstm_hidden'],
            config['model']['common_dim']
        )
        
        # Temporal encoder (Dilated CNN)
        self.temporal_encoder = self._build_temporal_encoder(config)
        self.temporal_projection = nn.Linear(
            config['model']['temporal_encoder']['output_dim'],
            config['model']['common_dim']
        )
        
        # Graph encoder (GAT)
        from torch_geometric.nn import GATConv
        self.graph_conv1 = GATConv(
            config['model']['graph_encoder']['input_dim'],
            config['model']['graph_encoder']['hidden_dim'],
            heads=config['model']['graph_encoder']['num_heads']
        )
        self.graph_conv2 = GATConv(
            config['model']['graph_encoder']['hidden_dim'] * config['model']['graph_encoder']['num_heads'],
            config['model']['common_dim'],
            heads=1
        )
        
    def _build_visual_cnn(self, config: Dict) -> nn.Module:
        """Build visual CNN backbone"""
        import torchvision.models as models
        
        backbone_name = config['model']['visual_encoder']['cnn_backbone']
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            # Remove final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def _build_temporal_encoder(self, config: Dict) -> nn.Module:
        """Build temporal encoder with dilated convolutions"""
        layers = []
        in_channels = config['model']['temporal_encoder']['input_dim']
        
        for i, dilation in enumerate(config['model']['temporal_encoder']['dilations']):
            out_channels = config['model']['temporal_encoder']['hidden_dim']
            layers.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation
            ))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        text_input: Optional[Dict] = None,
        visual_input: Optional[torch.Tensor] = None,
        temporal_input: Optional[torch.Tensor] = None,
        graph_input: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all encoders
        
        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            visual_input: Tensor of shape (batch, seq_len, channels, height, width)
            temporal_input: Tensor of shape (batch, seq_len, features)
            graph_input: Tuple of (node_features, edge_index)
        
        Returns:
            Dictionary of encoded representations
        """
        encodings = {}
        
        # Text encoding
        if text_input is not None:
            text_output = self.text_encoder(**text_input)
            text_emb = text_output.last_hidden_state[:, 0, :]  # CLS token
            encodings['text'] = self.text_projection(text_emb)
        
        # Visual encoding
        if visual_input is not None:
            batch_size, seq_len = visual_input.shape[:2]
            # Reshape for CNN processing
            visual_flat = visual_input.view(-1, *visual_input.shape[2:])
            visual_features = self.visual_cnn(visual_flat)
            visual_features = visual_features.view(batch_size, seq_len, -1)
            # LSTM processing
            lstm_out, _ = self.visual_lstm(visual_features)
            encodings['visual'] = self.visual_projection(lstm_out[:, -1, :])
        
        # Temporal encoding
        if temporal_input is not None:
            # Transpose for Conv1d: (batch, features, seq_len)
            temporal_t = temporal_input.transpose(1, 2)
            temporal_features = self.temporal_encoder(temporal_t)
            temporal_features = temporal_features.squeeze(-1)
            encodings['temporal'] = self.temporal_projection(temporal_features)
        
        # Graph encoding
        if graph_input is not None:
            node_features, edge_index = graph_input
            x = F.relu(self.graph_conv1(node_features, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.graph_conv2(x, edge_index)
            # Global pooling
            encodings['graph'] = torch.mean(x, dim=0, keepdim=True)
        
        return encodings


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for multi-modal fusion"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.common_dim = config['model']['common_dim']
        self.num_components = 3  # awareness, monitoring, control
        
        # Attention parameters for each modality and component
        self.modalities = ['text', 'visual', 'temporal', 'graph']
        self.components = ['awareness', 'monitoring', 'control']
        
        # Learnable attention weights
        self.attention_weights = nn.ParameterDict({
            f"{modality}_{component}": nn.Parameter(torch.randn(self.common_dim))
            for modality in self.modalities
            for component in self.components
        })
        
    def forward(self, encodings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical attention to fuse modalities
        
        Args:
            encodings: Dictionary of encoded representations
        
        Returns:
            Dictionary of fused representations for each component
        """
        fused = {}
        
        for component in self.components:
            component_encodings = []
            attention_scores = []
            
            for modality in self.modalities:
                if modality in encodings:
                    # Compute attention score
                    weight_key = f"{modality}_{component}"
                    score = torch.matmul(
                        encodings[modality],
                        self.attention_weights[weight_key]
                    )
                    attention_scores.append(score)
                    component_encodings.append(encodings[modality])
            
            if component_encodings:
                # Softmax over attention scores
                attention_scores = torch.stack(attention_scores, dim=1)
                attention_weights = F.softmax(attention_scores, dim=1)
                
                # Weighted sum of encodings
                component_encodings = torch.stack(component_encodings, dim=1)
                fused[component] = torch.sum(
                    attention_weights.unsqueeze(-1) * component_encodings,
                    dim=1
                )
        
        return fused


class MetacognitiveStateEstimator(nn.Module):
    """Estimator for metacognitive states"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        common_dim = config['model']['common_dim']
        
        # Component-specific estimators
        self.awareness_estimator = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.monitoring_estimator = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.control_estimator = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fused_representations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Estimate metacognitive states
        
        Args:
            fused_representations: Fused representations for each component
        
        Returns:
            Dictionary of metacognitive state estimates
        """
        states = {}
        
        if 'awareness' in fused_representations:
            states['awareness'] = self.awareness_estimator(fused_representations['awareness'])
        
        if 'monitoring' in fused_representations:
            states['monitoring'] = self.monitoring_estimator(fused_representations['monitoring'])
        
        if 'control' in fused_representations:
            states['control'] = self.control_estimator(fused_representations['control'])
        
        return states


class SRLFramework(nn.Module):
    """Complete Adaptive Multi-Modal AI Framework for Self-Regulated Learning"""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__()
        
        # Load configuration
        if config is None:
            if config_path is None:
                raise ValueError("Either config_path or config must be provided")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        
        # Initialize components
        self.encoder = MultiModalEncoder(config)
        self.attention = HierarchicalAttention(config)
        self.state_estimator = MetacognitiveStateEstimator(config)
        
    def forward(
        self,
        text_input: Optional[Dict] = None,
        visual_input: Optional[torch.Tensor] = None,
        temporal_input: Optional[torch.Tensor] = None,
        graph_input: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete framework
        
        Args:
            text_input: Text data
            visual_input: Visual data
            temporal_input: Temporal data
            graph_input: Graph data
        
        Returns:
            Dictionary of metacognitive state estimates
        """
        # Encode all modalities
        encodings = self.encoder(
            text_input=text_input,
            visual_input=visual_input,
            temporal_input=temporal_input,
            graph_input=graph_input
        )
        
        # Apply hierarchical attention
        fused = self.attention(encodings)
        
        # Estimate metacognitive states
        states = self.state_estimator(fused)
        
        return states
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

