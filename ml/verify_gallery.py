"""Comprehensive verification of classifier model on all gallery demo images."""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ml.config import CONFIG
from ml.models import get_classifier
import onnxruntime as ort

# Load fine-tuned PyTorch model
ckpt_file = CONFIG.checkpoint_dir / 'best_classifier_v3_finetuned.pt'
if not ckpt_file.exists():
    ckpt_file = CONFIG.checkpoint_dir / 'best_classifier_v3.pt'
ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
model = get_classifier(2)
model.load_state_dict(ckpt['model_state'])
model.eval()
threshold = ckpt.get('best_threshold', 0.45)
print(f'Model Checkpoint: {ckpt_file.name}')
print(f'Calibrated Threshold: {threshold:.2f}')

# Load ONNX session
onnx_path = CONFIG.output_classifier_onnx
sess = ort.InferenceSession(str(onnx_path))
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

demo_dir = CONFIG.demo_dir

print('\n=== CLASS 1 (Oil Spill) Gallery Images ===')
oil_correct = 0
oil_total = 0
for n in range(1, 11):
    for prefix in [f'class_1_{n:02d}.jpg', f'class_1_{n}.jpg']:
        p = demo_dir / prefix
        if not p.exists():
            continue
        oil_total += 1
        img = Image.open(p).convert('L').resize((400, 400))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Proportional cross-polarization scaling (vh = vv * 0.85)
        tensor = np.stack([arr, arr * 0.85], axis=0)[np.newaxis, ...].astype(np.float32)
        
        # ONNX run
        raw_out = sess.run([out_name], {in_name: tensor})[0]
        logit = float(raw_out.flat[0])
        prob = 1.0 / (1.0 + np.exp(-max(-20, min(20, logit))))
        decision = 'OIL' if prob >= threshold else 'CLEAN'
        ok = 'OK' if decision == 'OIL' else 'WRONG'
        if ok == 'OK':
            oil_correct += 1
        print(f'  {prefix:15s}: P={prob:.4f} -> {decision} [{ok}]')

print(f'Oil Recall: {oil_correct}/{oil_total} ({(oil_correct/max(1, oil_total))*100:.1f}%)')

print('\n=== CLASS 0 (Clean) Gallery Images ===')
clean_correct = 0
clean_total = 0
for n in range(1, 11):
    for prefix in [f'class_0_{n:02d}.jpg', f'class_0_{n}.jpg']:
        p = demo_dir / prefix
        if not p.exists():
            continue
        clean_total += 1
        img = Image.open(p).convert('L').resize((400, 400))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = np.stack([arr, arr * 0.85], axis=0)[np.newaxis, ...].astype(np.float32)
        
        raw_out = sess.run([out_name], {in_name: tensor})[0]
        logit = float(raw_out.flat[0])
        prob = 1.0 / (1.0 + np.exp(-max(-20, min(20, logit))))
        decision = 'OIL' if prob >= threshold else 'CLEAN'
        ok = 'OK' if decision == 'CLEAN' else 'WRONG'
        if ok == 'OK':
            clean_correct += 1
        print(f'  {prefix:15s}: P={prob:.4f} -> {decision} [{ok}]')

# User clean scene test
user_clean = demo_dir / 'user_clean_scene.jpg'
if user_clean.exists():
    clean_total += 1
    img = Image.open(user_clean).convert('L').resize((400, 400))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = np.stack([arr, arr * 0.85], axis=0)[np.newaxis, ...].astype(np.float32)
    raw_out = sess.run([out_name], {in_name: tensor})[0]
    logit = float(raw_out.flat[0])
    prob = 1.0 / (1.0 + np.exp(-max(-20, min(20, logit))))
    decision = 'OIL' if prob >= threshold else 'CLEAN'
    ok = 'OK' if decision == 'CLEAN' else 'WRONG'
    if ok == 'OK':
        clean_correct += 1
    print(f'  {user_clean.name:15s}: P={prob:.4f} -> {decision} [{ok}]')

print(f'Clean Specificity: {clean_correct}/{clean_total} ({(clean_correct/max(1, clean_total))*100:.1f}%)')
print('\nVerification complete.')
