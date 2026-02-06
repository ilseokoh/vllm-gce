# Google Cloud Compute Engine에서 vLLM 구동 학습자료

강좌영상의 내용을 Google Cloud Compute Engine에서 구현해보는 학습자료. 

## [강좌영상](https://www.youtube.com/watch?v=G85cWhD-Y5k&list=PLJkjrxxiBSFBYC3redRVJUc051ohXRQq0&index=1) 
  
## GCE 생성 
* us-central1 지역에 a2-highgpu-1g (12 vCPUs, 85 GB Memory) 머신 - A100 GPU Quota 가 1개 
* 모델 다운로드, vLLM, 도커 이미지 등 큰 용량의 다운로드를 위해 Boot Disk를 100G로 늘린다. 

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


### 2. Python 가상환경 생성 
```
git clone https://github.com/ilseokoh/vllm-gce.git
cd vllm-gce
uv sync
```

