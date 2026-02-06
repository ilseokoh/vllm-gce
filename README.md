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


### 2. Python 가상환경 생성 
```
git clone https://github.com/ilseokoh/vllm-gce.git
cd vllm-gce
uv sync
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