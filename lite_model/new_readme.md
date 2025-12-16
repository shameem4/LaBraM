# LaBraM System Map

This document explains how EEG data moves through the LaBraM codebase and
describes the responsibility of each major file. Use it as the first stop
when planning training, fine-tuning, or deployment changes.

## End-to-End Data & Training Flow

```text
Raw EEG (EDF/BDF/CNT ...)
   │
   ├─► dataset_maker/*.py  → HDF5 shards aligned to LaBraM conventions
   │       (channel pruning, filtering, resampling, windowing)
   │
   ├─► data_processor/ & utils.build_pretraining_dataset()
   │       Wrap HDF5 chunks as PyTorch Dataset objects + channel metadata
   │
   ├─► Training track 1: Tokenizer (VQ-NSP)
   │       run_vqnsp_training.py → engine_for_vqnsp.train_one_epoch()
   │           ↳ modeling_vqnsp.VQNSP enc/dec + norm_ema_quantizer codebook
   │           ↳ Produces checkpoints/checkpoints/vqnsp.pth
   │
   ├─► Training track 2: Masked EEG Modeling (MEM) pre-train
   │       run_labram_pretraining.py → engine_for_pretraining.train_one_epoch()
   │           ↳ Loads tokenizer, builds masked targets via get_codebook_indices
   │           ↳ modeling_pretrain.NeuralTransformerForMEM learns contextual
   │             representations; checkpoints saved to checkpoints/labram-*
   │
      └─► Fine-tuning
            run_class_finetuning.py → engine_for_finetuning.train_one_epoch()
               ↳ utils.prepare_TUAB/TUEV → PyTorch datasets + metrics
               ↳ modeling_finetune.NeuralTransformer consumes raw EEG patches
               ↳ Saves downstream heads (classification / regression)

```

## End-to-End Inference / Deployment Flow

```text
Raw EEG stream or archives (EDF/BDF/CNT, sensor packets)
   │
   ├─► Lightweight preprocessing
   │       • Apply same channel pruning/filtering/resampling rules as training
   │       • Segment into 200-sample patches (sliding window for streaming)
   │
   ├─► Dataset wrapper
   │       • utils.prepare_*() for offline eval (TUAB/TUEV, etc.)
   │       • Custom Dataset for streaming sensors (ensure channel ordering)
   │
   ├─► Model bootstrap
   │       • Instantiate modeling_finetune.LaBraM variant via timm.create_model()
   │       • Load checkpoint (finetuned or pretrained) with utils.load_state_dict()
   │       • Call model.eval(); optionally torch.jit.trace / torch.onnx.export
   │
   ├─► Execution paths
   │       • Offline metrics: run_class_finetuning.py --eval --finetune ckpt →
   │             engine_for_finetuning.evaluate() for BCE/CE metrics
   │       • Batch inference: simple script loads Dataset/DataLoader,
   │             loops over batches, collects logits/probabilities
   │       • Streaming/mobile: feed rolling windows (B×N×A×T) through
   │             TemporalConv front-end, keep channel map fixed, optionally
   │             cache hidden states for overlap-save style latency reduction
   │
   └─► Outputs
           • Classification logits / probabilities per window
           • Optional intermediate features via model.forward_features()
           • Serialized artifacts: TorchScript, ONNX, or CoreML for deployment
```

### Control Signals & Decisions

1. **Argument parsing (run_*.py):** CLI flags define hardware, dataset, and
   model hyperparameters. Each script builds models via `timm.create_model()`
   and data pipelines before delegating to its engine.
2. **Engines (engine_for_*.py):** Encapsulate the training loop, gradient
   scaling, logging, masking logic, EMA updates, and distributed hooks.
3. **Models (modeling_*.py):** Provide reusable Transformer blocks, tokenizer
   heads, and projection layers. They expose helper methods like
   `get_codebook_indices()` or `forward_features()` that the engines call.
4. **Utilities (utils.py):** Handle dataset construction, distributed setup,
   TensorBoard logging, metrics, cosine schedulers, checkpoint I/O, and
   channel-order bookkeeping so models receive consistent `input_chans`.

## File Guide

<!-- markdownlint-disable MD013 -->

| File / Folder              | Purpose                                                      |
| -------------------------- | ------------------------------------------------------------ |
| run_vqnsp_training.py      | CLI entry point for tokenizer training                       |
| engine_for_vqnsp.py        | Training/eval loops for VQ-NSP tokenizer                     |
| modeling_vqnsp.py          | VQNSP encoder-decoder and quantizer wiring                   |
| run_labram_pretraining.py  | MEM pre-training launcher with tokenizer checkpoint          |
| engine_for_pretraining.py  | Random masking, teacher-token generation, AMP for MEM        |
| modeling_pretrain.py       | NeuralTransformerForMEM with TemporalConv, PatchEmbed        |
| run_class_finetuning.py    | Downstream fine-tuning/eval entry point                      |
| engine_for_finetuning.py   | Training/eval steps for classification/regression heads      |
| modeling_finetune.py       | Core Transformer encoder with attention and classifier head  |
| utils.py                   | Distributed setup, dataset prep, schedulers, checkpoint I/O  |
| optim_factory.py           | Optimizer constructors and parameter grouping                |
| norm_ema_quantizer.py      | EMA-based vector quantizer for VQ-NSP codebook               |
| data_processor/            | Dataset definitions and preprocessing scripts                |
| dataset_maker/             | Raw-to-HDF5 conversion scripts and EEG utilities             |
| checkpoints/               | Default location for pretrained weights                      |

<!-- markdownlint-enable MD013 -->

## Working With the Flow

- **Swap datasets:** Modify the dataset lists in `run_labram_pretraining.py`
  or `run_vqnsp_training.py`, then update `time_window` and `stride_size` so
  sequence lengths stay near 256 tokens per sample.
- **Change model capacity:** Adjust the `--model` flag (registered in
  `modeling_*`) or add new configs there; the run scripts pick them up
  automatically through `timm.create_model()`.
- **Add downstream tasks:** Implement dataset prep + metrics in `utils.py`,
  then extend `get_dataset()` inside `run_class_finetuning.py`.
- **Inspect checkpoints:** Use `utils.load_state_dict()` to partially load or
  strip heads; the helper already drops mismatched classifier layers when
  `nb_classes` differs.

Refer back to this map whenever you introduce a new dataset, adjust model
shape, or rewire the tokenizer flow—the listed files outline where code
changes should land.
