## DeepSeek-V3-0324 Expert Pruning

This part of repo is built on [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) for [expert pruning](https://arxiv.org/abs/2410.12013) of DeepSeek-V3-0324 models. If you have any installation issues, please check the original repo.

The following code requires an 8×H200 GPU server for execution.

```bash
git clone https://github.com/Bobchenyx/CC-MoE.git
cd CC-MoE/DeepSeek-V3-Pruning/inference

# Convert the HF checkpoint into a format suitable for multi-GPU inference
python3 convert.py --hf-ckpt-path "<Path-To-DeepSeek-V3-0324>" \
                   --save-path "<Path-To-DeepSeek-V3-0324-DS-E256MP8>" \
                   --n-experts 256 --model-parallel 8

# Run inference to track expert usage & scores across all tokens
torchrun --nnodes 1 --nproc-per-node 8 generate.py \
         --ckpt-path "<Path-To-DeepSeek-V3-0324-DS-E256MP8>" \
         --config configs/config_671B.json --track-expert-scores

# Prune low-importance experts
python3 moe_pruner.py --input-hf-path "<Path-To-DeepSeek-V3-0324>" \
                      --input-expert-scores config_671B_expert_scores_seqlen_8k.json \
                      --prune-expert "<Num-Expert-To-Prune>" \
                      --output-hf-path "<Path-To-Pruned-FP8-Model>"

# Cast the pruned FP8 model back to BF16 for quantization
python3 fp8_cast_bf16.py --input-fp8-hf-path "<Path-To-Pruned-FP8-Model>" \
                         --output-bf16-hf-path "<Path-To-Pruned-BF16-Model>"
```

## Models
You may alternatively try our released pruned models:

DeepSeek-V3-0324-MoE-Pruner-E192 [🤗 Hugging Face](https://huggingface.co/tflsxyy/DeepSeek-V3-0324-MoE-Pruner-E192-bf16)

DeepSeek-V3-0324-MoE-Pruner-E160 [🤗 Hugging Face](https://huggingface.co/tflsxyy/DeepSeek-V3-0324-MoE-Pruner-E160-bf16)


## Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{xie2024moe,
  title={MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router},
  author={Xie, Yanyue and Zhang, Zhi and Zhou, Ding and Xie, Cong and Song, Ziang and Liu, Xin and Wang, Yanzhi and Lin, Xue and Xu, An},
  journal={arXiv preprint arXiv:2410.12013},
  year={2024}
}
```

## Acknowledgement

This repo benefits from [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), Thanks for their wonderful works.
