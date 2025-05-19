
# Controlled Image Generation with Stable Diffusion & ControlNet

This repository provides a comprehensive and reproducible pipeline for training, fine-tuning, and deploying a **ControlNet-augmented Stable Diffusion** model, specifically designed for **semantic segmentation–conditioned image synthesis**. Leveraging the **Cityscapes dataset**, the pipeline enables precise control over the image generation process by conditioning on structured scene layouts. It builds upon the publicly available [lllyasviel/sd-controlnet-seg](https://huggingface.co/lllyasviel/sd-controlnet-seg) checkpoint and implements the ControlNet framework proposed in [*Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543). By integrating segmentation masks as control signals, this setup facilitates high-resolution, photorealistic generation of complex urban environments with semantic alignment and spatial consistency. This work aims to serve both as a foundation for further research in controllable generative models and as a practical tool for urban scene synthesis.

<br>

## 📂 Repository Structure

Clone the pretrained pipeline from Hugging Face:

```bash
git clone https://huggingface.co/doguilmak/cityscapes-controlnet-sd15
```

Folder contents:

```
cityscapes-controlnet-sd15/
├── controlnet/                         # ControlNet weights
├── diffusion_pytorch_model-001.safetensors  # Core UNet weights
├── feature_extractor/                  # ControlNet vision encoder
├── model_index.json                    # Pipeline metadata & config
├── scheduler/                          # Scheduler configs (e.g. DDIM, PNDM)
├── text_encoder/                       # CLIP text encoder weights
├── tokenizer/                          # CLIP tokenizer files
├── unet/                               # U-Net denoiser architecture
└── vae/                                # VAE encoder & decoder

```

<br>

## 🛠️ Installation

```bash
git clone https://huggingface.co/doguilmak/cityscapes-controlnet-sd15
pip install diffusers transformers accelerate safetensors datasets # optional
```

_(Optional: use Conda)_

```bash
conda create -n controlnet-env python=3.8
conda activate controlnet-env

```

<br>

## 🚀 Usage Overview

### 1. Training (Fine-tuning ControlNet)

Training involves fine-tuning only the ControlNet-specific parameters while keeping the core Stable Diffusion components frozen. This approach allows the model to learn how to condition image generation on segmentation maps without altering the pre-trained generative capacity of the base model. The paired Cityscapes dataset (RGB image and segmentation map) provides strong spatial structure, making it ideal for tasks like urban scene synthesis. The training process leverages a simple noise prediction objective in the diffusion framework, enabling controllable and high-fidelity image outputs aligned with semantic layouts.

-   **Data**: Each input image $x_0$ is paired with a segmentation map $c$, resized to $256\times256$, normalized to $[-1, 1]$.
    
-   **Model Setup**: Load `sd-controlnet-seg` and freeze Stable Diffusion components (text encoder, UNet, VAE). Trainable layers are in ControlNet only.
    
- **Optimization**: AdamW optimizer with a learning rate of $1 \times 10^{-5}$, batch size 32, over 50 epochs (3475 samples), trained on an NVIDIA A100 40GB GPU (~2 hrs).
    
- **Noise Schedule**: Linear $\beta$ schedule from $\beta_1 = 0.0001$ to $\beta_T = 0.02$.
    
- **Noise Modeling**:  
  $x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
     
- **Loss Function**:  
  The loss function is defined as the expected squared difference between the predicted noise $\epsilon_\theta$ and the actual noise $\epsilon$ at each time step $t$, conditioned on the input image $x_t$, segmentation map $c$, and any additional context $y$. The only parameters optimized during training are those of the ControlNet, denoted as $\theta$.
  
-   **Checkpoints**: Saved every 5 epochs to `output/checkpoints/`, and exported as `full_pipeline/` at completion.

### 📄 **Usage Guide**:  
To quickly get started with the model and see how it works in action, refer to the [**`Usage.ipynb`**](/usage/Usage.ipynb) notebook. This notebook provides an easy-to-follow walkthrough for using the trained model for segmentation-guided image generation. It covers everything from loading the model to performing inference and generating high-quality images based on input segmentation maps. You can easily run the notebook to see how the model performs and make adjustments as needed.

In addition, you can find the **Cityscapes dataset color codes** for building your own input images and generating scenes [here](https://docs.cvat.ai/docs/manual/advanced/formats/format-cityscapes/).

<br>

### 2. Sampling & Inference

After training, use the `full_pipeline/` to perform segmentation-guided generation. For reproducible sampling, seed PyTorch and CUDA:

```python
import torch
from diffusers import set_seed
set_seed(42)
torch.cuda.manual_seed_all(42)
```

Load and configure the pipeline:

```python
from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "/content/cityscapes-controlnet-sd15/full_pipeline",
    safety_checker=None,
    torch_dtype=torch.float32
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_xformers_memory_efficient_attention()
pipeline.to("cuda")
```

Prepare the segmentation mask and run inference:

```python
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
control = preprocess(Image.open("path/to/seg_map.png")).unsqueeze(0).to("cuda")

config = {
    "prompt": ["ultra-detailed city street at dusk, cinematic lighting"],
    "negative_prompt": ["low-resolution, artifacts, watermark"],
    "num_steps": 50,
    "guidance_scale": 9.0,
    "generator": torch.Generator("cuda").manual_seed(42)
}

output = pipeline(
    prompt=config["prompt"],
    negative_prompt=config["negative_prompt"],
    image=control,
    control_image=control,
    num_inference_steps=config["num_steps"],
    guidance_scale=config["guidance_scale"],
    generator=config["generator"],
    output_type="pil"
)
output.images[0].save("output/inference.png")
```

**Tips**:

-   Set `num_images_per_prompt > 1` for multi-sample outputs.
    
-   Override `height` and `width` for custom sizes.
    
-   Enable `pipeline.enable_attention_slicing()` to reduce VRAM usage.

<br>

## 📸 Sample Outputs

Epoch 50 Samples

![Epoch 50](/samples/samples_50.png)

Inference Result

![Inference](/usage/inference.png)

<br>

## Limitations

-   The model was trained on **256×256** resolution; higher-resolution inference may lead to artifacts unless resized inputs are used.
    
-   It performs best on scenes that resemble **urban environments**, such as city streets and buildings.
    
-   The input control image must closely resemble **Cityscapes segmentation formats** (classes and layout).

<br>

## 📖 References

-   [lllyasviel/sd-controlnet-seg](https://huggingface.co/lllyasviel/sd-controlnet-seg)
    
-   Y. Zhao _et al._, “Adding Conditional Control to Text-to-Image Diffusion Models,” [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
