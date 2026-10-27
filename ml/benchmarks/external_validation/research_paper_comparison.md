# Research Paper Comparison — SpillSense vs. Published Approaches

## Table 1: Published Literature-Reported Results

> **Note**: These are results as reported in the original publications. They were obtained using each study's own dataset, splits, preprocessing, and evaluation protocol. Direct comparison across studies requires careful attention to these methodological differences.

| Study / Model | Year | Dataset | Test Definition | Task | Dice | IoU | Precision | Recall | F1 | Params | Latency |
|---------------|------|---------|-----------------|------|------|-----|-----------|--------|----|--------|---------|
| Zhu et al. (DeepLabV3 + hard negatives) | 2021 | Deep-SAR SOS (original) | Their test split | Seg. | N/A | 0.769 (mIoU) | N/A | N/A | N/A | ~40M | N/A |
| Trujillo-Acatitla et al. (CNN-UNet two-stage) | 2023 | Zenodo 8346860 | Their test split | Seg. + Cls. | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Standard U-Net (Ronneberger et al.) | 2015 | Various | Various | Seg. | ~0.75-0.85 | ~0.60-0.75 | N/A | N/A | N/A | ~31M | ~50-200ms |
| DeepLabV3+ (Chen et al.) | 2018 | Various | Various | Seg. | ~0.80-0.90 | ~0.70-0.82 | N/A | N/A | N/A | ~40-60M | ~100-300ms |
| SegFormer (Xie et al.) | 2021 | Various | Various | Seg. | ~0.82-0.88 | ~0.70-0.80 | N/A | N/A | N/A | ~3.7-84M | ~20-150ms |
| Garcia-INPE TinyUNet | 2024 | Cantarell (173 triplets) | 22 test scenes | Seg. | 0.7538 | 0.6049 | 0.9678 | 0.6173 | 0.7538 | ~2.1M | N/A |
| Garcia-INPE Hybrid (Felz+RF) | 2024 | Cantarell (173 triplets) | 22 test scenes | Seg. | 0.6289 | 0.4587 | 0.7693 | 0.5319 | 0.6289 | N/A | N/A |
| Garcia-INPE Felzenszwalb (test) | 2024 | Cantarell (173 triplets) | 22 test scenes | Seg. | 0.5449 | 0.3745 | 0.5484 | 0.5414 | 0.5449 | N/A | N/A |
| Garcia-INPE Felzenszwalb (full) | 2024 | Cantarell (173 triplets) | 173 full dataset | Seg. | 0.4525 | 0.2924 | 0.5756 | 0.3728 | 0.4525 | N/A | N/A |
| Yang & Singha (DARTIS/NRT system) | 2024 | Eastern Mediterranean | Their test scenes | Det. | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Notes on Published Results:
1. **Zhu et al. (2021)** reports mIoU (mean IoU across oil + background classes) = 0.769 on their Deep-SAR SOS test set. This is a multiclass metric including background IoU, **not directly comparable** to our binary oil-only IoU of 0.7227.
2. **Standard U-Net / DeepLabV3+ / SegFormer** ranges are compiled from multiple SAR and general segmentation studies — not from a single controlled experiment on the same oil spill dataset.
3. **Garcia-INPE baselines** — exact numeric results were read directly from `pipeline_comparison.csv` in the official repository.
4. **Trujillo-Acatitla et al.** — the Zenodo 8346860 paper describes a two-stage approach but specific benchmark numbers were not extracted from the publication in this audit.

---

## Table 2: SpillSense Reported Reference Results (Internal Validation)

> **These are our own results on our own training/validation data.** They are NOT independent external validation results.

| Model | Checkpoint | Dataset | Split | Task | Dice | IoU | Precision | Recall | F1 | Params | ONNX Size |
|-------|-----------|---------|-------|------|------|-----|-----------|--------|----|--------|-----------|
| SpillSegNet (production) | best_segmenter_v3_exp1_lr1e4.pt | Zenodo 8346860 | Internal val | Seg. | 0.8390 | 0.7227 | 0.8013 | 0.8804 | 0.8390 | 1,593,089 | 6.37 MB |
| DualPolOilSpillNet (production) | best_dualpol_classifier.pt | Zenodo 8346860 | Internal val | Cls. | N/A | N/A | N/A | 0.3145 | 0.4712 | 317,057 | 1.29 MB |
| DualPolOilSpillNet (staged) | best_dualpol_classifier_exp2_sampling_sw30.pt | Curated val | Curated val | Cls. | N/A | N/A | 0.9915 | 0.9624 | 0.9767 | 317,057 | N/A |

### Important Caveats:
- The production classifier's low recall (31.45%) on the baseline validation set reflects a different evaluation protocol than the staged classifier's curated validation.
- The staged classifier's high metrics (F1=0.9767) are on a **curated** validation set, not the standard independent test set.
- The segmenter's Dice of 0.8390 is a **micro Dice** across all pixels from the internal validation scenes.

---

## Table 3: SpillSense External Dataset Experimental Results

> **These results are populated ONLY from actual completed inference.** Results marked "PENDING" have not finished execution.

| Model | Dataset | Evaluation Type | Num Scenes | Micro Dice | Micro IoU | Micro Prec | Micro Rec | Pixel Accuracy | Notes |
|-------|---------|-----------------|------------|------------|-----------|------------|-----------|----------------|-------|
| SpillSegNet | D: Cantarell (test split) | Adapted (RGB-SAR to dual-pol synth) | 22 | 0.0976 | 0.0513 | 0.0518 | 0.8328 | 0.9241 | High recall (83.3%), low precision (5.2%) due to domain shift |
| SpillSegNet | D: Cantarell (full dataset) | Adapted (RGB-SAR to dual-pol synth) | 173 | 0.1569 | 0.0851 | 0.0871 | 0.7861 | 0.9312 | Consistent high recall across all 173 tiles |
| SpillSegNet | A: SOS (val split) | Adapted (single-pol to dual-pol synth) | 1,615 | 0.5056 | 0.3383 | 0.8890 | 0.3532 | 0.9572 | Strong precision (88.9%), moderate recall (35.3%) on diverse PALSAR/S1 SAR patches |
| SpillSegNet | B: DARTIS | BLOCKED | 0 | N/A | N/A | N/A | N/A | N/A | Metadata-only; no cropped patches available |
| SpillSegNet | C: CSIRO | BLOCKED | 0 | N/A | N/A | N/A | N/A | N/A | No segmentation dataset found on CSIRO portal |
| SpillSegNet | E: Zenodo S-1 | EXCLUDED (training data) | N/A | N/A | N/A | N/A | N/A | N/A | Used to train model; cannot serve as external benchmark |

---

## Methodological Warnings

1. **No direct zero-shot evaluation was possible** on any external dataset because no publicly available dataset provides dual-polarization (VV+VH) SAR patches in a format directly matching our model's input requirements (2-channel, Sigma0 dB normalized GeoTIFF at 512x512).

2. **Adapted evaluations** use channel synthesis (single grayscale channel duplicated + offset) to approximate dual-pol input. This is a significant methodological limitation that must be disclosed in any publication.

3. **Do NOT compare external adapted results directly with internal reference results** — the evaluation protocols, input data quality, and domain characteristics are fundamentally different.

4. **mIoU vs. binary IoU** — published results often report mean IoU across all classes (including background), which inflates the number compared to foreground-only IoU. Our IoU of 0.7227 is oil-class-only IoU.

## References

1. Zhu, Q., et al. (2021). "Oil Spill Contextual and Boundary-Supervised Detection Network Based on Marine SAR Images." IEEE TGRS. DOI: 10.1109/TGRS.2021.3115492
2. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
3. Chen, L.-C., et al. (2018). "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV.
4. Xie, E., et al. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS.
5. Yang, Y.-J. & Singha, S. (2024). "A near real-time automated oil spill detection and early warning system using Sentinel-1 SAR imagery." IJRS, 45(6), 1997-2027.
6. Garcia-INPE (2024). Cantarell Oil Spill/Seepage Segmentation Dataset. Zenodo DOI: 10.5281/zenodo.22214144.
7. Trujillo-Acatitla, R., et al. (2023). "Sentinel-1 SAR Oil spill image dataset." Zenodo DOI: 10.5281/zenodo.8346860.
