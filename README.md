
![Inference](/assets/cover.png)

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

Training involves fine-tuning only the **ControlNet-specific parameters**, while keeping the core Stable Diffusion components (**text encoder, UNet, VAE**) frozen. This design allows the model to learn how to control image generation via semantic segmentation maps without disrupting the pretrained generative capacity of the base model.

The training is performed on the **Cityscapes dataset**, where each image is paired with its corresponding segmentation mask. These pairs provide strong spatial structure for guiding controllable synthesis of complex urban scenes.

The model is optimized using a simple noise prediction objective from the diffusion framework, aiming to learn the mapping from segmentation-guided latent noise to clean latent image representations.

#### 🧾 Training Details

- **Data**:  
  Each RGB image $x_0$ is paired with a segmentation map $c$. Both are resized to $256 \times 256$ and normalized to the $[-1, 1]$ range, using the following transformation:

  <p align="center">
      <img src="https://quicklatex.com/cache3/17/ql_c6e5f5931464c654a08192ba803aeb17_l3.png" alt="Normalization Formula">
  </p>

- **Model Setup**:  
  The pipeline is initialized using the `sd-controlnet-seg` checkpoint. All core components of Stable Diffusion (VAE, UNet, and text encoder) are frozen during training. Only the ControlNet layers are trainable.

- **Noise Schedule**:  
  A **linear β-schedule** is used with:

  <p align="center">
      <img src="https://quicklatex.com/cache3/de/ql_22ba62b963482b4bf42f2301467e9bde_l3.png" alt="Noise Schedule">
  </p>

- **Noise Modeling**:  
  The model predicts added noise $\epsilon$ in the diffusion process, defined as:

  <p align="center">
      <img src="https://quicklatex.com/cache3/cf/ql_a88fb8e1286e4469485772e48750d1cf_l3.png" alt="Noise Modeling">
  </p>

- **Loss Function**:  
  The training loss is the **Mean Squared Error (MSE)** between the predicted and actual noise at each timestep:

  <p align="center">
      <img src="https://quicklatex.com/cache3/ca/ql_5fadb370c243b362b47e6e21c163e7ca_l3.png" alt="Loss Function">
  </p>

  where $\epsilon_\theta$ is the predicted noise, $\epsilon$ is the actual noise sampled from a standard normal distribution, and $\theta$ are the trainable parameters of ControlNet. The loss is normalized by the number of gradient accumulation steps.

- **Optimization**:  
  The model is optimized using the **AdamW optimizer** with a **learning rate of $3.38 \times 10^{-8}$**. Training was performed with a **batch size of 32** and **gradient accumulation over 8 steps**, which effectively simulates a larger batch size of 256. Gradient clipping was applied with a **maximum norm of 1.0** to stabilize training. The model was trained for **50 epochs** using **3,475 image-mask pairs**, completing in approximately 2 hours on an **NVIDIA A100 40GB GPU**.

- **Sampling Configuration during Evaluation**:
  - **NUM_STEPS**: 50  
    → Controls the number of denoising steps used during inference. More steps generally produce better quality.
  - **GUIDANCE**: 9.0  
    → Classifier-free guidance scale for balancing fidelity and diversity.

- **Final Loss**:  
  After 50 epochs of training, the final training loss reached $0.0201$.
  
-   **Checkpoints**: Saved every 10 epochs to `output/checkpoints/`, and exported as `full_pipeline/` at completion.

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

### 📄 **Usage Guide**  
To quickly get started with the model and see how it works in action, refer to the [**`Usage.ipynb`**](/usage/Usage.ipynb) notebook. This notebook provides an easy-to-follow walkthrough for using the trained model for segmentation-guided image generation. It covers everything from loading the model to performing inference and generating high-quality images based on input segmentation maps. You can easily run the notebook to see how the model performs and make adjustments as needed.

In addition, you can find the **Cityscapes dataset color codes** for building your own input images and generating scenes [here](https://docs.cvat.ai/docs/manual/advanced/formats/format-cityscapes/).

**Tips**:

-   Set `num_images_per_prompt > 1` for multi-sample outputs.
    
-   Override `height` and `width` for custom sizes.
    
-   Enable `pipeline.enable_attention_slicing()` to reduce VRAM usage.

<br>

## 📸 Sample Outputs

These examples illustrate the model’s ability to generate photorealistic urban scenes guided by semantic segmentation maps. The outputs demonstrate strong spatial alignment between the input masks and the synthesized content, capturing realistic lighting, structure, and urban textures.

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

-   [Stable diffusion ControlNet segmentation HuggingFace page.](https://huggingface.co/lllyasviel/sd-controlnet-seg)
    
-   Y. Zhao _et al._, “Adding Conditional Control to Text-to-Image Diffusion Models,” [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)

-   [Can Michael Hucko's 'sar2rgb' Repository](https://github.com/canmike/sar2rgb)
