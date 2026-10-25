"""UPAR 2027 Track 1 building blocks.

upar.data      annotations, image cache, dataset, augmentation
upar.metrics   mA / Instance-F1 / Challenge_Avg
upar.losses    LOSSES registry (bce, logit_adj, focal, asl) with shared smoothing / weighting / masking
upar.models    HEADS registry (linear, query) + build_model()

Kept import-free on purpose: the submission vendors upar.models without pandas / torchvision.
"""
