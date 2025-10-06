<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->
# DeepSeek-V3-0324 Expert Pruning

This part of repo is built on [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) for [expert pruning](https://arxiv.org/abs/2410.12013) of DeepSeek-V3-0324 models. If you have any installation issues, please check the original repo.

## Expert Pruning

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

## Quantization

The quantization process is following [unsloth](https://unsloth.ai/blog/deepseek-v3-0324).

```bash
git clone git@github.com:unslothai/llama.cpp.git
apt-get update
apt-get install build-essential cmake curl libcurl4-openssl-dev -y
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-bench llama-gguf-split llama-quantize llama-imatrix llama-server
cp llama.cpp/build/bin/llama-* llama.cpp
```
```bash
mkdir /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf/
python3 llama.cpp/convert_hf_to_gguf.py /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-bf16 --outfile /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf --outtype bf16 --split-max-size 50G
./llama.cpp/llama-imatrix -m /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf-00001-of-00022.gguf -f expert_scores_json/mmlu.txt -o expert_scores_json/e192_imatrix.dat
mkdir /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S/
./llama.cpp/llama-quantize --imatrix expert_scores_json/e192_imatrix.dat --keep-split /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf/DeepSeek-V3-0324-MoE-Pruner-E192-bf16-gguf-00001-of-00022.gguf /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S.gguf IQ1_S
```

## Models

DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S [🤗 Hugging Face](https://huggingface.co/tflsxyy/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S)

DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S [🤗 Hugging Face](https://huggingface.co/tflsxyy/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S)

## Execution

On a 4xV100 server:

```bash
./llama.cpp/llama-cli --model /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S/DeepSeek-V3-0324-MoE-Pruner-E192-IQ1_S-00001-of-00022.gguf  --cache-type-k q8_0 --threads 64 --n-gpu-layers 61 -no-cnv --prio 3 --temp 0.3 --min_p 0.01 --ctx-size 4096 --seed 3407 --prompt "<｜User｜>How are you doing?<｜Assistant｜>"
```

## Evaluation

The [llama_server.py](https://github.com/tflsxyy/lm_eval_gguf/blob/main/llama_server.py) is a patch code for evaluating local gguf by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) that is much faster than [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Start a local server by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server).

```bash
./llama.cpp/llama-server -m /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S-00001-of-00018.gguf -ngl 62
```

Use lm-eval to evaluate the model in another terminal.

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
HF_DATASETS_TRUST_REMOTE_CODE=1 lm-eval --model llama-server --tasks arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,piqa,rte,winogrande --model_args base_url=http://127.0.0.1:8080
```

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

This repo benefits from [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), [llama.cpp](https://github.com/ggml-org/llama.cpp), and [unsloth](https://unsloth.ai/blog/deepseek-v3-0324). Thanks for their wonderful works.
