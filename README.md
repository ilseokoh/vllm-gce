# Google Cloud Compute Engine에서 vLLM 구동 학습자료

강좌영상의 내용을 Google Cloud Compute Engine에서 구현해보는 학습자료. 

## 강좌영상

* [강좌영상](https://www.youtube.com/watch?v=G85cWhD-Y5k&list=PLJkjrxxiBSFBYC3redRVJUc051ohXRQq0&index=1) 
* [문서](https://cafe.naver.com/xxxjjhhh/373)
  
## GCE 생성 
* us-central1 지역에 a2-highgpu-1g (12 vCPUs, 85 GB Memory) 머신 - A100 GPU Quota 가 1개 
* 모델 다운로드, vLLM, 도커 이미지 등 큰 용량의 다운로드를 위해 Boot Disk를 **200G**로 늘린다. 

## Nvidia 드라이버 및 CUDA 설치 

[관련문서](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=ko#install-script)를 참조하여 아래 내용 실행 

1. 설치 스크립트 다운로드 
1. 설치스크립트 실행시 lspci 가 없다는 에러가 발생하면 pciutils 설치 
```bash
sudo apt-get install -y pciutils 
```
1. GPU가 있는지 확인 
```bash
lspci | grep -i nvidia
```
1. 드라이버 설치 → reboot 
1. CUDA 설치 → reboot 
1. 확인 
```bash
nvidia-smi
```

## 실습 

### 1. HuggingFace 에서 Llama 3.1 8B 모델 다운로드 

* [HuggingFace](https://huggingface.co/) 가입
* Access Token 생성 (Read Permission)
* [Llama-3.1-8B-Instruct 모델](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 다운로드 
   1. git-lfs 설치 
   ```
   sudo apt update
   sudo apt install git-lfs
   ```
   1. Clone - username / access token 사용
   ```
   cd ~ 
   mkdir models 
   cd models
   git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   ```
   1. pip / uv 설치
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   sudo apt install python3-pip
   ```
   1. 소스코드 Clone
   ```
   cd ~ 
   git clone https://github.com/ilseokoh/vllm-gce.git
   cd vllm-gce
   uv sync
   ```


### 2. vLLM 테스트 

1. test.py 에서 다운로드한 모델의 위치 변경 
```
# 모델 위치
model_path = "/home/user_id/models/Meta-Llama-3.1-8B-Instruct/"

def main():
```
1. test.py 실행
```
cd ~ 
cd vllm-gce
uv run test.py
```

**출력 예시**
```
INFO 02-06 03:33:17 [utils.py:261] non-default args: {'disable_log_stats': True, 'enforce_eager': True, 'model': '/home/admin_iloh_altostrat_com/models/Meta-Llama-3.1-8B-Instruct'}
INFO 02-06 03:33:17 [model.py:541] Resolved architecture: LlamaForCausalLM
INFO 02-06 03:33:17 [model.py:1561] Using max model len 131072
INFO 02-06 03:33:17 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 02-06 03:33:17 [vllm.py:624] Asynchronous scheduling is enabled.
WARNING 02-06 03:33:17 [vllm.py:662] Enforce eager set, overriding optimization level to -O0
INFO 02-06 03:33:17 [vllm.py:762] Cudagraph is disabled under eager mode
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:18 [core.py:96] Initializing a V1 LLM engine (v0.15.1) with config: model='/home/admin_iloh_altostrat_com/models/Meta-Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='/home/admin_iloh_altostrat_com/models/Meta-Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/home/admin_iloh_altostrat_com/models/Meta-Llama-3.1-8B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': True}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:23 [parallel_state.py:1212] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://10.128.0.16:35995 backend=nccl
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:23 [parallel_state.py:1423] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:24 [gpu_model_runner.py:4033] Starting to load model /home/admin_iloh_altostrat_com/models/Meta-Llama-3.1-8B-Instruct...
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:26 [cuda.py:364] Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.36s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:02<00:02,  1.42s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:04<00:01,  1.45s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:04<00:00,  1.04s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:04<00:00,  1.18s/it]
(EngineCore_DP0 pid=10479) 
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:31 [default_loader.py:291] Loading weights took 4.75 seconds
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:32 [gpu_model_runner.py:4130] Model loading took 14.99 GiB memory and 6.720924 seconds
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:33 [gpu_worker.py:356] Available KV cache memory: 19.28 GiB
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:33 [kv_cache_utils.py:1307] GPU KV cache size: 157,936 tokens
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:33 [kv_cache_utils.py:1312] Maximum concurrency for 131,072 tokens per request: 1.20x
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:33 [core.py:272] init engine (profile, create kv cache, warmup model) took 1.71 seconds
(EngineCore_DP0 pid=10479) WARNING 02-06 03:33:34 [vllm.py:669] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore_DP0 pid=10479) INFO 02-06 03:33:34 [vllm.py:762] Cudagraph is disabled under eager mode
INFO 02-06 03:33:35 [llm.py:343] Supported tasks: ['generate']
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 237.34it/s]
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:26<00:00, 26.22s/it, est. speed input: 0.76 toks/s, output: 39.06 toks/s]
 
1. 해리포터와 마법왕 (Harry Potter and the Philosopher's Stone)
   - 해리 포터는 11살 때 부모를 잃고 오일버트 던스에게 입양된다.
   - 해리는 자신의 정체성을 발견하고, 마법학교인 호그와트魔法학교에 입학한다.
   - 해리는 친구들인 론과 헤르미온느와 함께 필ософ스의 돌을 찾기 위해 모험을 떠난다.

2. 해리포터와 비밀의 방 (Harry Potter and the Chamber of Secrets)
   - 해리는 두 번째 학년이 시작되면서 새로운 미스터리가 발생한다.
   - 해리는 자신이 처음부터 호그와트에서 있었던 것이 아니라는 것을 알게되고, 그 사실을 밝히기 위해 노력한다.
   - 해리는 친구들과 함께 비밀의 방을 찾아내어, 그 안에 있는 몬스터를 물리친다.

3. 해리포터와 아즈카반의 죄수 (Harry Potter and the Prisoner of Azkaban)
   - 해리는 세 번째 학년이 시작되면서, 신비한 교사들이 나타나고, 아즈카반의 죄수가 탈출한다.
   - 해리는 아즈카반의 죄수를 잡아야 하는 임무를 맡는다.
   - 해리는 친구들과 함께 아즈카반의 죄수를 잡아내고, 그의 진실을 알아낸다.

4. 해리포터와 불사조 기사단 (Harry Potter and the Goblet of Fire)
   - 해리는 네 번째 학년이 시작되면서, 호그와트에서 대회가 열린다.
   - 해리는 불사조 기사단의 일원이 되고, 대회에서 우승하지만, 실제로는 다른 사람의 이름으로 참가했다는 사실을 밝힌다.
   - 해리는 친구들과 함께 대회를 끝내고, 볼드모트의 복귀를 막으려 한다.

5. 해리포터와 아시안의 귀환자 (Harry Potter and the Order of the Phoenix)
   - 해리는 다섯 번째 학년이 시작되면서, 볼드모트가 다시 활동하기 시작한다.
   - 해리는 친구들과 함께 볼드모트를 막기 위해 노력한다.
   - 해리는 Dumbledore의 Army를 만들고, 볼드모트를 막기 위한 계획을 수립한다.

6. 해리포터와 반지의 비밀 (Harry Potter and the Half-Blood Prince)
   - 해리는 여섯 번째 학년이 시작되면서, 볼드모트가 계속해서 활동한다.
   - 해리는 볼드모트의 과거를 조사하고, 그의 약점을 알아낸다.
   - 해리는 친구들과 함께 볼드모트를 막기 위해 노력한다.

7. 해리포터와 죽음의 성물 (Harry Potter and the Deathly Hallows)
   - 해리는 일곱 번째 학년이 시작되면서, 볼드모트가 마지막으로 등장한다.
   - 해리는 친구들과 함께 볼드모트를 막기 위해 노력한다.
   - 해리는 볼드모트를 물리치고, 호그와트의 위기를 극복한다.

해리포터 시리즈는 마법과 사랑, 친구애 대한 이야기로 구성되어 있으며, 어린이와 청소년들에게 많은 영향을 미쳤다. 

해리포터 시리즈는 다음과 같은 요소들을 포함하고 있다.

- 마법과 상상력: 해리포터 시리즈는 마법과 상상력을 통해 이야기를 풀어나간다.
- 친구애 대한 사랑: 해리포터 시리즈는 친구애 대한 사랑과 동료애를 강조한다.
- 희생과 용기: 해리포터 시리즈는 희생과 용기를 통하여 이야기를 진행한다.
- 악과 선: 해리포터 시리즈는 악과 선의 대결을 통해 이야기를 풀어나간다.

해리포터 시리즈는 어린이와 청소년들에게 많은 영향을 미쳤으며, 어른들도 읽어보면 좋습니다. 

해리포터 시리즈는 다음과 같은 장르를 포함하고 있다.

- 판타지
- 어드벤처
- 로맨스

해리포터 시리즈는 다음과 같은 수상과 명예를 받았다.

- 뉴욕 타임스 베스트셀러 목록
- 영국 베스트셀러 목록
- 미국 베스트셀러 목록

해리포터 시리즈는 다음과 같은 영화화와 애니메이션화가 이루어졌다.

- 해리포터 영화 시리즈
- 해리포터 애니메이션 시리즈

해리포터 시리즈는 다음과 같은 책으로 구성되어 있다.

- 해리
```




------------
gcloud compute ssh --project=kevin-ai-playground --zone=us-central1-f instance-20260205-004951
ssh -i ~/.ssh/google_compute_engine admin_iloh_altostrat_com@34.55.88.60

sudo docker build --no-cache -t test_vllm_1:latest .


docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker

docker run -d --gpus all --shm-size=16G -v /mnt/disks/vllm/llama3:/app/models -e MODEL_PATH="/app/models/Llama-3.1-8B-Instruct/" -p 8080:8080 test_vllm_1

### 3. FastAPI 연동

1. server.py 에서 model 위치 변경 
```
app = FastAPI() 

model_path = "/home/user_id/models/Meta-Llama-3.1-8B-Instruct/"
llm = LLM(model=model_path, gpu_memory_utilization=0.9,  tensor_parallel_size=1)
```
2. server 실행 
```
uv run uvicorn server:app --host 0.0.0.0 --port 8080
```
3. API 호출 테스트 
```
curl -X POST "http://localhost:8080/generate/" -H "Content-Type: application/json" -d '{"query": "해리포터의 줄거리를 한글로 간략히 설명해 주세요."}'
```


### 4. 비동기 치러 (AyncLLMEngine)

1. 
