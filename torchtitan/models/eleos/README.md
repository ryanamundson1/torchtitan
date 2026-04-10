# Eleos

> **Eleos** (Greek: ἔλεος — *mercy, compassion*) is a novel foundation LLM
> that integrates moral reasoning as an **architectural property**, not a
> post-hoc filter.  It is built on top of DeepSeek V3's Multi-Latent
> Attention and trained entirely within the [torchtitan](https://github.com/pytorch/torchtitan) framework.

---

## Architecture at a Glance

| Component | Description |
|---|---|
| **DualHemisphereAttention** | Splits MLA heads per-layer into a *logical* (analytical/deductive) and *generative* (creative/expressive) sub-stream — a learned left-brain / right-brain partition |
| **MoralGovernor** | Differentiable MLP inside every block that produces a moral score, a blend gate (α), and an adversarial-intent probability; these route the hemisphere outputs at every layer |
| **AdversarialDetector** | Sub-module of MoralGovernor that learns to flag manipulative inputs and shifts the blend gate toward deliberate, moral-elevating output |
| **Moral Auxiliary Loss** | `−mean(moral_scores) × moral_loss_weight` — a secondary gradient signal that rewards high moral alignment across all layers, trained jointly with cross-entropy |

The head-split ratio between hemispheres is **configurable per layer**, defaulting to a graduated pattern: more logical in early layers (token-level processing) transitioning to more generative in later layers (semantic composition).

---

## Available Sizes

| Flavor | Layers | Dim | Experts | Notes |
|---|---|---|---|---|
| `debugmodel` | 6 | 256 | 8 | Fast CI / dev iteration |
| `debugmodel_flex_attn` | 6 | 256 | 8 | Uses FlexAttention |
| `16B` | 27 | 2048 | 64 | Dense layer 0 + 26 MoE layers |
| `236B` | 60 | 5120 | 160 | LoRA Q projections |
| `671B` | 61 | 7168 | 256 | LoRA Q, sigmoid router |

---

## Tokenizer

Eleos uses a standard BPE tokenizer.  During early training you can reuse
the DeepSeek-compatible tokenizer:

```bash
# Download 16B-compatible tokenizer (for debug / 16B runs)
python scripts/download_hf_assets.py \
  --repo_id deepseek-ai/deepseek-moe-16b-base \
  --assets tokenizer
```

Place your own tokenizer at `./assets/hf/eleos-tokenizer/` for the 236B and 671B configs.

---

## Training

### Quick debug run (single GPU / CPU)

```bash
MODEL=eleos CONFIG=eleos_debugmodel ./run_train_mac.sh
```

### Debug run with FlexAttention

```bash
MODEL=eleos CONFIG=eleos_debugmodel_flex_attn ./run_train_mac.sh
```

### 16B model

```bash
# Download tokenizer first (see above), then:
MODEL=eleos CONFIG=eleos_16b ./run_train_mac.sh
```

### 236B model (multi-node)

```bash
MODEL=eleos CONFIG=eleos_236b ./run_train_mac.sh
```

### 671B model (large cluster)

```bash
MODEL=eleos CONFIG=eleos_671b ./run_train_mac.sh
```

---

## Configuration Options

### Moral Loss Weight

Controls how strongly the model is pushed toward high moral alignment during
training.  Default is `1e-3` (same magnitude as the MoE load-balance loss).

Override in Python:

```python
from torchtitan.models.eleos import model_registry, EleosModel

cfg = model_registry("16B").model
cfg.moral_loss_weight = 5e-4   # softer moral pressure
```

### Per-Layer Hemisphere Split

Each layer's fraction of heads devoted to logical reasoning is set
independently.  The named configs use a graduated pattern (higher logical
fraction in early layers), but you can supply arbitrary lists:

```python
from torchtitan.models.eleos import eleos_configs, EleosModel
from torchtitan.models.eleos import _build_eleos_layers

# Example: uniform 60/40 split across all 27 layers of the 16B config
fractions = [0.60] * 27
```

The `logical_head_fraction` for each layer is stored in:

```
model.layers["<n>"].attention.logical_head_fraction
```

### MoralGovernor Hyper-parameters

Per-layer attention config fields:

| Field | Default | Description |
|---|---|---|
| `moral_gate_hidden` | `dim // 8` | Hidden size of the governor MLP |
| `moral_override_alpha` | `0.85` | Blend weight toward logical when adversarial intent is detected |
| `adversarial_threshold` | `0.5` | Probability above which a token triggers the override |

---

## Checkpointing

Eleos uses TorchTitan's standard DCP (Distributed Checkpoint) format.
Checkpoints are saved every `CheckpointManager.Config(interval=N)` steps.

```bash
# Resume from a checkpoint directory
MODEL=eleos CONFIG=eleos_16b CHECKPOINT_DIR=./checkpoints/eleos_16b ./run_train.sh
```

---

## State Dict Adapter — Do You Need One?

**No — not for training from scratch.**

A `StateDictAdapter` (like the one in `deepseek_v3/`) is only needed when
**loading pre-trained HuggingFace weights** into TorchTitan's format.
Because Eleos has a novel architecture (dual Q projections, MoralGovernor
weights, etc.) there are no public HuggingFace checkpoints to load from.

You would need to write a `StateDictAdapter` only if you:

1. **Partially initialise** Eleos weights from a DeepSeek V3 checkpoint
   (shared KV matrices `wkv_a`, `kv_norm`, `wkv_b` and FFN/MoE weights are
   compatible; Q and output projections must be split/initialised fresh).
2. **Export** trained Eleos weights back to HuggingFace format for inference
   with `transformers`.

Until either of those use-cases applies, the standard TorchTitan DCP save/load
path works without any adapter.

---

## Adding Adversarial Training Data

The `AdversarialDetector` inside each `MoralGovernor` learns to recognise
harmful intent from the hidden-state distribution.  To sharpen this signal:

1. Include **contrast pairs** in your dataset: benign prompts paired with
   adversarial rewrites of the same prompt.
2. Mark adversarial sequences with a special token or dataset field so the
   positive signal can be amplified in the moral auxiliary loss.
3. Increase `moral_loss_weight` (try `5e-3`) when training on adversarial data
   to strengthen the internal moral compass.

---

## Inference & Generation

At a lower level, when running inference manually through Python, remember to disable the moral auxiliary loss return to get plain logits:

```python
model.eval()
with torch.no_grad():
    logits = model(tokens, return_moral_loss=False)
```

The moral governor still **operates during inference** — it shapes the blend of logical vs. generative attention at every layer. Only the auxiliary training loss computation is skipped.

### Quick Local Generation Check
To quickly test generating text natively against your saved configurations and checkpoints natively on macOS without crashing localhost communication limits, use the customized generation script:

```bash
./run_generate_mac.sh --prompt="What is the true meaning of artificial alignment?"
```

This runs a rapid forward pass over your output mesh distributions using TorchTitan's embedded `test_generate.py` evaluation suite.

### Exporting to External Inference Engines
TorchTitan acts primarily as a highly optimized distributed pre-training engine. For rigorous downstream tasks, conversational chats, or endpoint hosting, you are heavily encouraged to export the completed `torchtitan` distributed checkpoint into a format optimized for deployment engines (e.g. vLLM or Ollama).

To translate the sharded Torchtitan checkpoint into standard HuggingFace Safetensors format, run the native conversion helper:

```bash
python3 scripts/checkpoint_conversion/convert_to_hf.py \
    --checkpoint_dir outputs/checkpoint/ \
    --output_dir outputs/hf_conversion/
```

From HuggingFace format, standard frameworks like `llama.cpp` can compile the `safetensors` into quantized `.gguf` archives for Ollama ingestion.
