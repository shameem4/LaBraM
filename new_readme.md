# LaBraM System Map

This document explains how EEG data moves through the LaBraM codebase and
describes the responsibility of each major file. Use it as the first stop
when planning training, fine-tuning, or deployment changes.

## End-to-End Data, Training, and Inference Flow

```
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

   Deployment / inference:
      • run_class_finetuning.py --eval --finetune <checkpoint>
         ↳ Loads pretrained weights (teacher/student) via utils.load_state_dict()
         ↳ Builds DataLoader with same channel map
         ↳ engine_for_finetuning.evaluate() runs forward-only pass, aggregates metrics
      • For bespoke apps export modeling_finetune.NeuralTransformer via torch.jit
         or ONNX: instantiate model, load checkpoint, call model.eval(), then
         trace with sample tensor shaped [batch, channels, windows, patch]
      • Optional streaming: feed rolling windows through TemporalConv front-end
         and reuse cached hidden states if implemented downstream
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

| File / Folder | Purpose |
| --- | --- |
| run_vqnsp_training.py | CLI entry point for tokenizer training, builds datasets and launches `engine_for_vqnsp`.
| engine_for_vqnsp.py | Training / eval loops for the VQ-NSP tokenizer, including codebook usage reporting.
| modeling_vqnsp.py | Defines the `VQNSP` encoder-decoder, quantizer wiring, and `register_model` factory for timm.
| run_labram_pretraining.py | MEM pre-training launcher: loads tokenizer checkpoint, builds multi-montage datasets, configures optimizer + schedulers.
| engine_for_pretraining.py | Implements random masking, teacher-token generation, loss accumulation, AMP, and gradient accumulation for MEM.
| modeling_pretrain.py | Contains `NeuralTransformerForMEM` plus helper modules (TemporalConv, PatchEmbed). Outputs masked token predictions.
| run_class_finetuning.py | Downstream fine-tuning/eval entry. Handles dataset-specific metrics, optimizer grouping, EMA, and checkpoint loading.
| engine_for_finetuning.py | Batches training/eval steps for classification or regression heads with mixup-ready hooks.
| modeling_finetune.py | Core Transformer encoder used for tokenizer decoder, MEM, and downstream heads. Includes attention blocks, embeddings, and classifier head.
| utils.py | Shared helpers for distributed setup, dataset prep (TUAB/TUEV + generic), schedulers, metric logging, and checkpoint utilities.
| optim_factory.py | Optimizer constructors, parameter grouping (layer-wise lr decay), and custom schedulers.
| norm_ema_quantizer.py | EMA-based vector quantizer used by VQ-NSP to maintain the neural codebook.
| data_processor/ | Dataset definitions (`dataset.py`, preprocessing scripts) used by `utils.prepare_*`.
| dataset_maker/ | Raw-to-HDF5 conversion scripts (per dataset) plus shared EEG utilities for filtering & buffering.
| checkpoints/ | Default location for pretrained and tokenizer weights referenced by scripts.

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
