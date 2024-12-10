# Assigment 4 for ANLP

This repo contains an AWQ reproduction with a new search space compared to the original paper.

## Environment Setup

We recommend:

* At least 32GB GPU memory (e.g. AWS g4dn.2xlarge instance) for running AWQ on LLaMA-7B models
* 16GB GPU memory is sufficient for smaller models like OPT-1.3B and OPT-2.7B

## Prerequisties

1. Ensure CUDA drivers are installed
2. Follow guidelines from [AWQ](https://github.com/mit-han-lab/llm-awq?tab=readme-ov-file) repo
3. If `python setup.py` fails, set specific CUDA architecture:

```
TORCH_CUDA_ARCH_LIST="8.0" python setup.py install
```

**Note:** Running real quantized models with int4 weights requires hardware support.

## Useful commands

Download model from hugging face

```bash
huggingface-cli download meta-llama/Llama-2-7b --local-dir ~/models/model_weights/Llama-2-7b
```

Run AWQ

```bash
python -m awq.entry --model_path ~/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq ~/awq_cache_test/llama-2-7b-w4-g128.pt
```

Evaluate wikitext perplexity

```bash
python -m awq.entry --model_path ~/models/llama-2-7b \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq ~/awq_cache/llama-2-7b-w4-g128.pt \
    --q_backend fake
```

Running boolq task

```bash
python -m awq.entry --model_path ~/models/llama-2-7b \
    --tasks boolq \
    --w_bit 4 --q_group_size 128 \
    --load_awq ~/awq_cache/llama-2-7b-w4-g128.pt \
    --q_backend fake
```

## Additional Information

This repo uses lm-eval for benchmarking. For all supported tasks, refer to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
