import os
os.environ["SB_DISABLE_LAZY_IMPORT"] = "1"
import speechbrain as sb
from speechbrain.inference.classifiers import EncoderClassifier

sb_model_source = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
print(f"Attempting to load {sb_model_source} using EncoderClassifier.from_hparams...")

try:
    classifier = EncoderClassifier.from_hparams(source=sb_model_source)
    print("Success using EncoderClassifier.from_hparams!")
    # Test inference
    import torch
    import numpy as np
    signal = torch.randn(1, 16000)
    out_prob, score, index, text_lab = classifier.classify_batch(signal)
    print("Inference success:", text_lab)
except Exception as e:
    print(f"FAILED with: {type(e).__name__}: {e}")
