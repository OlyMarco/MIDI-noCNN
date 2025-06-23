import math
import numpy as np
import random

import torch
import torch.nn as nn
from former import FormerModel
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Former model: similar approach to "felix"
class MidiFormerRaw(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        # Token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # For deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # Linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), formerConfig.d_model)

        attn_config = BertConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        self.interaction_attn = BertAttention(attn_config)

        self.use_fif = use_fif

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm"):
        bs, slen, _ = input_ids.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        if not self.use_fif:
            # Convert input_ids into embeddings and merge them through linear layer
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]))
            embs = torch.cat([*embs], dim=-1)
        else:
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]).unsqueeze(2)) # B x L x 1 x d
            embs = torch.cat([*embs], dim=-2) # B x L x F x d

            embs_shape = embs.shape
            embs = embs.view(-1, embs_shape[2], embs_shape[3]) # (B x L) x F x d

            self_attention_outputs  = self.interaction_attn(embs)
            embs_interaction = self_attention_outputs[0]

            embs = embs_interaction.view(embs_shape[0], embs_shape[1], embs_shape[2], embs_shape[3]).reshape((embs_shape[0], embs_shape[1], embs_shape[2]*embs_shape[3]))


        emb_linear = self.in_linear(embs)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_ids.device), attn_mask], dim=1)

        # Feed to former
        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask,
                             output_hidden_states=output_hidden_states)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])

class MidiFormerNoAttn(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        # Token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # For deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # Linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), formerConfig.d_model)

        attn_config = BertConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        self.interaction_attn = BertAttention(attn_config)

        self.use_fif = use_fif
        
        # Add CNN structure, ported from MidiBert
        self.stack_cnn = StackTransitionCNN(input_channels=2, output_dim=self.hidden_size)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.fusion_layers = [2, 5, 8, 11]  # Select layers to fuse

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm", pctm=None, nltm=None):
        bs, slen, _ = input_ids.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        if not self.use_fif:
            # Convert input_ids into embeddings and merge them through linear layer
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]))
            embs = torch.cat([*embs], dim=-1)
        else:
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]).unsqueeze(2)) # B x L x 1 x d
            embs = torch.cat([*embs], dim=-2) # B x L x F x d

            embs_shape = embs.shape
            embs = embs.view(-1, embs_shape[2], embs_shape[3]) # (B x L) x F x d

            self_attention_outputs = self.interaction_attn(embs)
            embs_interaction = self_attention_outputs[0]

            embs = embs_interaction.view(embs_shape[0], embs_shape[1], embs_shape[2], embs_shape[3]).reshape((embs_shape[0], embs_shape[1], embs_shape[2]*embs_shape[3]))

        emb_linear = self.in_linear(embs)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_ids.device), attn_mask], dim=1)

        # Feed to former
        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask,
                        output_hidden_states=output_hidden_states)
        
        # # Process pctm and nltm, similar to logic in MidiBert
        # if pctm is not None and nltm is not None:
        #     # Handle NaN values
        #     pctm = torch.nan_to_num(pctm, nan=0.0)
        #     nltm = torch.nan_to_num(nltm, nan=0.0)
            
        #     # Ensure matrix shape is correct
        #     if len(pctm.shape) == 3:
        #         pctm = pctm.unsqueeze(1)  # [batch_size, 1, 12, 12]
        #     if len(nltm.shape) == 3:
        #         nltm = nltm.unsqueeze(1)  # [batch_size, 1, 12, 12]
            
        #     # Stack features along channel dimension
        #     stacked_features = torch.cat([pctm, nltm], dim=1)  # [B, 2, 12, 12]
            
        #     # Use CNN to process stacked features
        #     combined_features = self.stack_cnn(stacked_features)
            
        #     # Feature normalization
        #     combined_features = F.normalize(combined_features, dim=1)
            
        #     # Get hidden states list
        #     hidden_states = list(y.hidden_states)

        #     # Fuse features layer by layer
        #     for layer_idx in self.fusion_layers:
        #         if layer_idx >= len(hidden_states):
        #             continue
                
        #         # Get current layer representation
        #         layer_hidden = hidden_states[layer_idx]
                
        #         # Get special token (first token)
        #         special_token = layer_hidden[:, 0]
                
        #         # Fuse features
        #         fused_input = torch.cat([special_token, combined_features], dim=1)
        #         enhanced_features = self.feature_fusion(fused_input)
                
        #         # Expand matrix features to entire sequence
        #         _, seq_len, _ = layer_hidden.shape
        #         expanded_features = enhanced_features.unsqueeze(1).expand(-1, seq_len, -1)
        #         # Apply fusion to entire sequence
        #         layer_hidden = layer_hidden + expanded_features

        #         # Update current layer
        #         hidden_states[layer_idx] = layer_hidden
            
        #     # Update hidden states
        #     y.hidden_states = tuple(hidden_states)

        return y
    
    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])


import torch.nn.functional as F
class StackTransitionCNN(nn.Module):
    def __init__(self, output_dim, input_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Rest of structure same as TransitionCNN
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.layer_norm = nn.LayerNorm(32 * 3 * 3)
        
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Same forward pass as TransitionCNN
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        self.n_tokens = [len(e2w[key]) for key in e2w]
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        self.word_emb = nn.ModuleList([Embeddings(self.n_tokens[i], self.emb_sizes[i]) for i in range(len(e2w))])

        self.in_linear = nn.Linear(sum(self.emb_sizes), formerConfig.d_model)

        attn_config = BertConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        self.interaction_attn = BertAttention(attn_config)

        self.use_fif = use_fif
        
        self.stack_cnn = StackTransitionCNN(input_channels=2, output_dim=self.hidden_size)
        
        # Add attention fusion layer
        self.fusion_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=12)
        self.gate_linear = nn.Linear(self.hidden_size * 2, 1)
        # self.gate_net = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fusion_layers = [2,5,8,11]  # Select layers to fuse

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm", pctm=None, nltm=None):
        bs, slen, _ = input_ids.shape
        special_mark = torch.zeros((bs, 1)).long().to(input_ids.device) if mode == "mlm" else torch.ones((bs, 1)).long().to(input_ids.device)
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        if not self.use_fif:
            embs = torch.cat([self.word_emb[i](input_ids[..., i]) for i in range(len(self.e2w))], dim=-1)
        else:
            embs = torch.cat([self.word_emb[i](input_ids[..., i]).unsqueeze(2) for i in range(len(self.e2w))], dim=-2)
            embs_shape = embs.shape
            embs = embs.view(-1, embs_shape[2], embs_shape[3])
            self_attention_outputs = self.interaction_attn(embs)
            embs_interaction = self_attention_outputs[0]
            embs = embs_interaction.view(embs_shape[0], embs_shape[1], embs_shape[2] * embs_shape[3])

        emb_linear = self.in_linear(embs)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_ids.device), attn_mask], dim=1)

        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)

        # Delete CNN
        # if pctm is not None and nltm is not None and self.training:
        #     # print(f"pctm shape: {pctm.shape}, nltm shape: {nltm.shape}")
        #     pctm = torch.nan_to_num(pctm, nan=0.0)
        #     nltm = torch.nan_to_num(nltm, nan=0.0)
            
        #     if len(pctm.shape) == 3:
        #         pctm = pctm.unsqueeze(1)
        #     if len(nltm.shape) == 3:
        #         nltm = nltm.unsqueeze(1)
            
        #     stacked_features = torch.cat([pctm, nltm], dim=1)  # [batch_size, 2, 12, 12]
        #     combined_features = self.stack_cnn(stacked_features)  # [batch_size, hidden_size]
        #     combined_features = F.normalize(combined_features, dim=1)
            
        #     hidden_states = list(y.hidden_states)
            
        #     for layer_idx in self.fusion_layers:
        #         if layer_idx >= len(hidden_states):
        #             continue
                
        #         layer_hidden = hidden_states[layer_idx]  # [batch_size, seq_len + 1, hidden_size]
                
        #         # Use attention mechanism to fuse CNN features and each token's representation
        #         combined_features_expanded = combined_features.unsqueeze(0)  # [1, batch_size, hidden_size]
        #         layer_hidden_transposed = layer_hidden.transpose(0, 1)  # [seq_len + 1, batch_size, hidden_size]
        #         fused_output, _ = self.fusion_attention(
        #             query=combined_features_expanded,
        #             key=layer_hidden_transposed,
        #             value=layer_hidden_transposed
        #         )  # fused_output: [1, batch_size, hidden_size]
                
        #         fused_output = fused_output.squeeze(0)  # [batch_size, hidden_size]
        #         enhanced_features = fused_output.unsqueeze(1).expand(-1, slen + 1, -1)  # [batch_size, seq_len + 1, hidden_size]
                
        #         # Add fused features to original hidden layer features
        #         gate_input = torch.cat([layer_hidden, enhanced_features], dim=-1)
        #         gate = torch.sigmoid(self.gate_linear(gate_input))
        #         layer_hidden = gate * layer_hidden + (1 - gate) * enhanced_features
        #         # gate_mean = gate.mean().item()
        #         # gate_min = gate.min().item()
        #         # gate_max = gate.max().item()
        #         # print(f"Layer {layer_idx} gate stats: mean={gate_mean:.4f}, min={gate_min:.4f}, max={gate_max:.4f}")
        #         hidden_states[layer_idx] = layer_hidden
            
        #     y.hidden_states = tuple(hidden_states)

        return y
    
    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])