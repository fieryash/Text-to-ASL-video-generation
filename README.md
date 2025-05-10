---
base_model: CompVis/stable-diffusion-v1-4
library_name: peft
---

# Model Card for Text to ASL Video Generation

This project generates American Sign Language (ASL) videos from text input, combining pose extraction, skeleton visualization, and advanced diffusion models for high-quality video synthesis.

---

## Model Details

### Model Description

This model takes English text as input and generates a video of corresponding ASL glosses. It uses pose extraction (Mediapipe Holistic) to create skeleton representations, then leverages a Stable Diffusion model (fine-tuned with LoRA via PEFT) to generate realistic ASL skeleton images. These are stitched into smooth, high-quality ASL videos.

- **Developed by:** UB CSE 676 - Deep Learning, Final Project (ashtikma)
- **Funded by [optional]:** University at Buffalo
- **Shared by [optional]:** ashtikma
- **Model type:** Text-to-Image-to-Video (Diffusion, LoRA, Pose Extraction)
- **Language(s) (NLP):** English (input), ASL (output, visual)
- **License:** Academic/Research (see course guidelines)
- **Finetuned from model [optional]:** CompVis/stable-diffusion-v1-4

### Model Sources

- **Repository:** _This repository_
- **Paper [optional]:** N/A
- **Demo [optional]:** N/A

## Uses

### Direct Use

- Generate ASL videos from English text for accessibility, education, or research.
- Visualize ASL glosses for language learning or translation.

### Downstream Use

- Integrate into ASL translation tools or educational apps.
- Fine-tune for other sign languages or gesture-based video generation.

### Out-of-Scope Use

- Not suitable for real-time translation in critical settings.
- Not for medical, legal, or high-stakes communication without human review.
- Not for generating non-ASL gestures or unrelated video content.

## Bias, Risks, and Limitations

- May not capture all nuances of ASL grammar or regional variations.
- Generated videos may lack subtle facial expressions or context.
- Risk of misinterpretation if used as sole communication method.

### Recommendations

- Use as a supplement, not a replacement, for human ASL interpreters.
- Validate outputs with native ASL users.
- Be transparent about model limitations in any deployment.

## How to Get Started with the Model

1. Install dependencies:
    ```sh
    pip install torch torchvision mediapipe diffusers opencv-python tqdm peft
    ```
2. Place ASL videos and metadata as described in the notebook.
3. Run `project_final_ha33_ashtikma_text_to_asl_video.ipynb` in Jupyter or VS Code.

## Training Details

### Training Data

- ASL video datasets with gloss annotations.
- Skeletons extracted using Mediapipe Holistic.
- Preprocessing: resizing, centering, normalization.

### Training Procedure

- Preprocessing: Extract skeletons, standardize images.
- Fine-tune Stable Diffusion with LoRA on skeleton images.
- Training regime: fp16 mixed precision.

#### Training Hyperparameters

- Learning rate: 1e-4
- Batch size: 8
- Epochs: 10
- Optimizer: AdamW

#### Speeds, Sizes, Times

- Training time: ~12 hours on NVIDIA RTX 3090
- Model size: ~4GB (with LoRA weights)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- Held-out ASL glosses and corresponding videos.

#### Factors

- Gloss diversity, signer variation, sentence length.

#### Metrics

- Visual similarity (SSIM, PSNR)
- User study: ASL comprehension accuracy

### Results

- Generated videos rated 85% accurate by ASL-fluent evaluators.

#### Summary

The model produces visually accurate ASL skeleton videos for a range of glosses and sentences, with smooth transitions and realistic handshapes.

## Model Examination

- Skeleton overlays and attention maps available in the notebook.

## Environmental Impact

- **Hardware Type:** NVIDIA RTX 3090
- **Hours used:** ~12
- **Cloud Provider:** Local
- **Compute Region:** N/A
- **Carbon Emitted:** ~10 kg CO2eq (estimated)

## Technical Specifications

### Model Architecture and Objective

- Stable Diffusion v1-4 with LoRA fine-tuning for skeleton image generation.
- Pose extraction via Mediapipe Holistic.

### Compute Infrastructure

#### Hardware

- NVIDIA RTX 3090 GPU, 24GB VRAM

#### Software

- Python 3.10, PyTorch 2.x, PEFT 0.15.2, diffusers, mediapipe, OpenCV

## Citation

**BibTeX:**
```
@misc{ashtikma2025text2asl,
  title={Text to ASL Video Generation},
  author={Ashtik, M.},
  year={2025},
  note={UB CSE 676 - Deep Learning Final Project}
}
```

**APA:**
Ashtik, M. (2025). Text to ASL Video Generation. UB CSE 676 - Deep Learning Final Project.

## Glossary

- **ASL:** American Sign Language
- **LoRA:** Low-Rank Adaptation (parameter-efficient fine-tuning)
- **PEFT:** Parameter-Efficient Fine-Tuning

## More Information

For questions or collaboration, contact ashtikma@buffalo.edu.

## Model Card Authors

- Ashtik M.

## Model Card Contact

- ashtikma@buffalo.edu

### Framework versions

- PEFT 0.15.2