import os
from transformers import PatchTSTConfig, PatchTSTForPrediction

TSTconfig = PatchTSTConfig(
    num_input_channels = 1,
    context_length = 168,
    prediction_length = 24,
    patch_length = 24,
    patch_stride = 24,
    d_model = 256,
    num_attention_heads = 8,
    num_hidden_layers = 8,
    ffn_dim = 1024,
    dropout = 0.2,
    head_dropout = 0.2,
    pooling_type = None,
    channel_attention = False,
    scaling = "std",
    pre_norm = True,
    do_mask_input = False
)

model = PatchTSTForPrediction(TSTconfig)
model.save_pretrained(os.path.join("saved_models", "PatchTSTBackbone"))