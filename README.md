## Docker build (최초 1회)
```bash
git clone https://github.com/f56e751/iitp_perception.git
cd iitp_perception

# 모델 가중치 다운로드 (~1.2 GB, repo에 포함되지 않음)
wget -c --show-progress -O checkpoint_best.pth "https://www.dropbox.com/scl/fi/yzvoq6w4nm8x9dr5jr0ve/checkpoint_best.pth?rlkey=042faepbpcsc3liw2o3ak5oxz&st=284u74ef&dl=0"

# 베이스 이미지를 포함된 Dockerfile로 로컬 빌드 (외부 레지스트리에서 받아오지 않음).
# 태그 chaehyeonsong/grounded_sam 은 Dockerfile.local / docker.sh / run_one.sh 가
# 참조하므로, 이름을 바꾸려면 그쪽도 함께 수정해야 함.
docker build -t chaehyeonsong/grounded_sam .
```

위 도커는 pytorch:2.3.1-cuda12.1 환경에서 build됩니다. GPU driver가 cuda 12.1을 지원하지 않으면 Dockerfile의 베이스 태그를 더 낮춰서 사용해주세요.

## GroundingDINO CUDA 확장 빌드 (최초 1회, 필수)
```bash
docker run -it --gpus all --ipc=host -v $PWD:/mnt --name iitp chaehyeonsong/grounded_sam:latest
# ↑ 컨테이너 진입 후:
cd /mnt
python -m pip install --no-build-isolation -e grounding_dino
```
이 단계는 GroundingDINO의 CUDA 커널(`grounding_dino/groundingdino/_C*.so`)을 컴파일합니다.
GPU 추론에 **필수**이며, `.so`는 repo에 포함되지 않으므로(`.gitignore`) clone 후 머신마다 한 번 빌드해야 합니다.
(CPU에서는 순수 파이썬 fallback이 돌지만 이 프로젝트는 GPU 전제라 사실상 필수.)
빌드 결과 `.so`는 mount된 호스트 폴더(`grounding_dino/`)에 남으므로, 이후 `iitp_local`·eval 컨테이너에서도 재빌드 없이 재사용됩니다.

> 실제 perception 실행은 아래 **Local capture + MJPEG stream**(실시간 카메라) 또는 **Perception evaluation**(저장 이미지 평가) 섹션 참고.

(선택) 배치 모드 단독 확인 — 폴더 안 이미지들에 검출을 돌려 annotated 결과 저장:
```bash
python3 iitp_object_detector.py -i <폴더>/    # 결과: tmp_results/results_0.2_0.4/<n>.jpg
```
`-i` 폴더 안에는 `images/` 하위 폴더가 있어야 합니다. 예시 (`data_iitp_2/very_hard/` 같은 샘플 데이터는 repo에 미포함):
```bash
<폴더>
    └── images
        ├── 37.jpg
        ├── 38.jpg
        └── ...
```

## Exiting docker
```bash
# docker에서 나가기
exit

# 기존 docker 삭제
docker rm iitp
```

---

## 구조: 프로덕션 진입점 vs 평가 도구

- **`main.py`** — 프로덕션 진입점. 카메라 입력 → 모델 추론 → 결과를 `results_local/detections.jsonl`에
  기록 + 포트 8080 MJPEG 스트림 송출. 이것만 한다. 검출기는 `--backend`로 고른다
  (`sam3` 기본 / `dino`). 실행은 `run_sam3_local.sh` 또는 `docker_local.sh`(아래 "Local capture" 섹션).
- **`sam3_detector.py`** — SAM3 TensorRT 백엔드 어댑터. `SAM3-trt/` 런타임을 감싸서
  `object_detector`와 **같은** `(bounding_boxes, class_names, confidences)`를 돌려준다.
  tensorrt/pycuda import는 지연 로딩이라 이 모듈 자체는 어디서나 import 가능.
- **`SAM3-trt/`** — vendoring한 SAM3 TensorRT 런타임(수정 없음). 아래 "SAM3 TensorRT backend" 참고.
- **`iitp_object_detector.py`** — GroundingDINO 엔진(라이브러리). `--backend dino`와 `scripts/`가 import.
  단독 폴더 배치 검출도 가능: `python iitp_object_detector.py -i <폴더>/`.
- **`scripts/`** — 오프라인 평가·분석 도구. 각 스크립트를 직접 실행하거나 `scripts/run_one.sh`로 묶어서 실행.

| `scripts/` 스크립트 | 설명 | 실행 환경 |
|---|---|---|
| `capture_only.py` | 카메라 캡처만 (검출 없이 프레임 저장) | 카메라 (`iitp_local`) |
| `eval_detector.py` | 저장 이미지에 클래스별 점수 평가 (`scores.csv`) | base 이미지 |
| `group_by_object.py` | `scores.csv` 검출을 객체별 CSV로 클러스터링 | 순수 파이썬 |
| `group_by_track.py` | 검출을 이동 트랙으로 클러스터링 | 순수 파이썬 |
| `analyze_tracks.py` | 트랙 예측 vs 정답 라벨 비교 | 순수 파이썬 |
| `apply_corrections.py` | 수동 라벨 보정 적용 | 순수 파이썬 |
| `docker_capture.sh` | `capture_only.py`를 USB 패스스루 컨테이너로 실행 | 카메라 (`iitp_local`) |
| `run_one.sh` | capture → eval → group 전체 파이프라인 | — |

---

## SAM3 TensorRT backend (기본 검출기)

`main.py --backend sam3`은 GroundingDINO 대신 SAM3를 TensorRT engine으로 돌린다.
**송출 계약은 그대로다**: 포트 8080의 `/stream`, `/detections`, `/detections/stream`,
`/latency`, jsonl 스키마, 클래스 이름(`transparent`/`metal`/`cardboard`)이 모두 동일하므로
로봇 PC의 `perception_client.py`는 수정 없이 동작한다. 바뀌는 것은 박스를 만드는 모델뿐이다.

GroundingDINO는 박스를 직접 회귀하지만 SAM3는 프롬프트별 **마스크**를 내고 그 마스크의
bounding box를 쓴다. 그래서 박스가 물체 윤곽에 더 붙는다. 프롬프트는 `SAM3-trt`의
`DEFAULT_PROMPT_SPECS`(카테고리당 1~2개 자연어 문구)를 그대로 쓴다.

### 구성 파일
- `SAM3-trt/serve_trt_iitp_usls.py` — vendoring한 런타임. `UslsSam3Runner`(vision/text/decoder
  engine 3개) + `RuntimeSegmenter`(카테고리 충돌 해소 · NMS · 인스턴스화)
- `sam3_detector.py` — 위 런타임을 이 repo의 검출 계약으로 변환하는 얇은 어댑터
- `download_sam3_onnx.sh` / `wrap_sam3_trt.sh` — ONNX 내려받기 / TensorRT engine 빌드
- `run_sam3_local.sh` — 카메라 + SAM3 + :8080 실행 런처 (conda env를 USB 패스스루 컨테이너에 마운트)

### 사전 준비 (최초 1회)

**1) 추론 환경.** TensorRT 런타임은 grounded_sam 컨테이너(torch 2.3 / CUDA 12.1)와 공존하기
어려워서 호스트에 별도 conda env를 둔다. 실행은 그 env를 `iitp_local` 컨테이너에
bind-mount해서 한다 — 이 호스트에는 librealsense udev 규칙이 없고 카메라 USB 노드가
root 소유(`crw-rw-r-- root root`)라, 호스트에서 바로 열면 `No device connected`가 난다.
컨테이너는 오직 USB 패스스루 용도이고 파이썬/모델은 전부 conda env 것을 쓴다.

```bash
# miniforge (이미 있으면 생략)
curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash mf.sh -b -p /PublicSSD/iitp/sam3_env/miniforge3
/PublicSSD/iitp/sam3_env/miniforge3/bin/conda create -y -n sam3 python=3.11

PIP=/PublicSSD/iitp/sam3_env/miniforge3/envs/sam3/bin/pip
$PIP install "numpy<2.3" opencv-python-headless pillow tokenizers supervision pyrealsense2
$PIP install torch --index-url https://download.pytorch.org/whl/cpu   # SAM3-trt import용, 추론엔 미사용
$PIP install transformers
# 드라이버가 CUDA 12.x면 반드시 cu12 빌드. 그냥 `pip install tensorrt`를 쓰면
# CUDA 13 빌드(tensorrt_cu13)가 깔려 engine 생성이 실패한다.
$PIP install "tensorrt-cu12<11"
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH $PIP install pycuda
```

**2) ONNX 다운로드 (~4.2 GB)** — `weights/`는 gitignore 대상이라 clone 후 한 번 받아야 한다.

```bash
cd /PublicSSD/iitp/iitp_perception
bash download_sam3_onnx.sh runtime
```

**3) TensorRT engine 빌드 (~10분).** engine은 GPU/CUDA/TensorRT/드라이버 버전에 묶여 있으므로
머신마다 다시 만들어야 한다.

```bash
PATH=/PublicSSD/iitp/sam3_env/miniforge3/envs/sam3/bin:$PATH bash wrap_sam3_trt.sh
```

산출물:
```text
weights/SAM3-trt/usls_engines_b2p2/vision_b2_fp16.engine
weights/SAM3-trt/usls_engines/text_b6_fp16.engine
weights/SAM3-trt/usls_engines/decoder_b1_p32_fp16.engine
weights/usls/tokenizer.json
```

`Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed`가 뜨면 TensorRT
버전이 바뀐 것이다. 같은 ONNX로 `wrap_sam3_trt.sh`만 다시 돌리면 된다.

### 실행

```bash
cd /PublicSSD/iitp/iitp_perception
./run_sam3_local.sh
# 옵션은 그대로 전달된다: ./run_sam3_local.sh --port 8081
```
- engine 로드(컨테이너 기동 포함 ~10초) 후 `streams ready on :8080 ...` 이 뜨면 준비 완료
- 중단: `Ctrl+C`
- GroundingDINO로 되돌리려면 기존 경로 그대로: `./docker_local.sh`

### 튜닝

임계값·프롬프트는 `sam3_detector.default_args()`에 모여 있고, 기본값은 vendoring한
`serve_trt_iitp_usls.py`의 argparse 기본값 + `--precision-tuned-preset`과 같다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `score_thr` | 0.4 | decoder raw score 1차 컷 |
| `transparent/metal/cardboard_min_score` | 0.60 / 0.50 / 0.45 | 클래스별 최종 컷 (preset) |
| `cross_category_iou` / `_overlap` | 0.25 / 0.55 | 같은 물체를 두 클래스가 물었을 때 해소 기준 |
| `merge_nms_iou` | 0.5 | 클래스 내부 박스 NMS |
| `process_area_thr` | 1000 | 마스크 분할/병합 판단 면적(px) |
| `temporal_vote` | `False` | 프레임 간 마스크 투표. 벨트 속도(px/frame)를 재고 나서 켤 것 |

---

## Local capture + MJPEG stream (robot6 직결 모드)

RealSense 카메라를 이 서버(robot6)에 USB로 직결해서, 네트워크 왕복 없이 바로 SAM 추론을 돌리고 바운딩박스가 그려진 영상을 다른 컴퓨터(로봇 PC 등)의 브라우저로 실시간 송출하는 모드.

### 구성 파일
- `Dockerfile.local` — 기존 `chaehyeonsong/grounded_sam` 이미지 위에 `pyrealsense2`만 얹은 파생 이미지 (`iitp_local:latest`)
- `docker_local.sh` — USB 패스스루(`--privileged`, `-v /dev:/dev`) + `--network host`로 컨테이너 실행
- `main.py` — pyrealsense2로 컬러+깊이 캡처 → 검출 백엔드(`--backend sam3|dino`) 호출 →
  `results_local/detections.jsonl`에 결과 추가, 포트 8080에서 영상/검출결과 송출
- `run_sam3_local.sh` — SAM3 TensorRT 백엔드로 실행(호스트 conda env + USB 패스스루 컨테이너)
- `streaming.py` — :8080 HTTP 전송 계층(MJPEG 영상 + 검출결과 JSON). stdlib 전용
- `scripts/recv_detections.py` — 스트림 동작 빠른 점검용 CLI
- `scripts/perception_client.py` — 로봇 PC repo로 복사해 쓰는 재접속 클라이언트(라이브러리)
- `scripts/fake_stream.py` — 카메라/모델 없이 합성 검출을 송출하는 더미 producer(수신 테스트용)

### 사전 준비 (최초 1회)
```bash
# 1) iitp 계정에 docker 그룹 추가 (sudo 필요)
sudo usermod -aG docker iitp
# 로그아웃/재로그인 또는 newgrp docker

# 2) 파생 이미지 빌드 (베이스 이미지 chaehyeonsong/grounded_sam:latest가 이미 로컬에 있어야 함)
cd /PublicSSD/iitp/iitp_perception
docker build -f Dockerfile.local -t iitp_local:latest .

# 3) RealSense 카메라를 robot6 USB 3.x 포트에 연결
lsusb | grep RealSense        # Intel Corp. Intel(R) RealSense(TM) ... 확인
```

### 서버 실행 (robot6)
```bash
# SAM3 TensorRT (기본)
cd /PublicSSD/iitp/iitp_perception && ./run_sam3_local.sh

# GroundingDINO (기존 경로)
cd /PublicSSD/iitp/iitp_perception && ./docker_local.sh
```
- 모델 로드(SAM3 ~10초 / DINO ~20초) 후 `streams ready on :8080 ...` 출력되면 준비 완료
- 중단: 콘솔에서 `Ctrl+C` (컨테이너는 `--rm`이라 자동 정리)

SSH 끊어도 계속 돌리고 싶을 때:
```bash
cd /PublicSSD/iitp/iitp_perception && nohup ./run_sam3_local.sh > stream.log 2>&1 &
# 중단: kill %1  (docker 경로라면 docker stop iitp_local)
```

### 다른 컴퓨터에서 받기 (포트 8080 공유)
같은 LAN이면 서버 IP `147.46.175.15` (안 되면 `147.46.240.59`) 사용. 여러 명 동시 접속 가능.

| 엔드포인트 | 용도 | 소비 방법 |
|---|---|---|
| `GET /stream` | annotated MJPEG 영상 | 브라우저 / VLC |
| `GET /detections` | 최신 검출 결과 1건 (JSON) | 폴링 / 디버그 |
| `GET /detections/stream` | **실시간 검출 스트림 (NDJSON)** | 한 줄당 JSON 1개, 프레임마다 push |
| `GET /latency` | 서버 수신/송신 시각 | 로봇 PC의 RTT·시계 오프셋·잔여 스트림 지연 측정 |

빠른 확인 (스트림 동작 점검용):
```bash
python3 scripts/recv_detections.py --url http://147.46.175.15:8080/detections/stream
```

카메라/모델 없이 수신 코드를 테스트하려면 **더미 producer**를 띄운다 (stdlib만, GPU 불필요):
```bash
python3 scripts/fake_stream.py --port 8080            # 합성 검출 레코드를 :8080으로 발행
# 다른 터미널/PC에서:
python3 scripts/perception_client.py --url http://127.0.0.1:8080/detections/stream
```

**로봇 제어 PC**는 이 repo를 클론하지 말고, `scripts/perception_client.py` 를 자기 repo로
**복사**해서 사용 (의존성 stdlib뿐, 자동 재접속 + 스키마 버전 경고 포함):
```python
from perception_client import stream_detections   # 로봇 repo에 복사한 파일

def on_record(rec):
    for bbox, cls, conf in zip(rec["bounding_boxes"], rec["class_names"], rec["confidences"]):
        if conf < 0.3:
            continue
        # bbox = [top-left, top-right, bottom-right, bottom-left]
        # 각 점은 belt-plane [X, Y, Z] 좌표(m)
        ...

stream_detections("http://147.46.175.15:8080/detections/stream", on_record)
```
이 repo는 카메라 PC에만 두고, 로봇 PC는 위 와이어 스키마(계약)에만 의존하는 게 권장 구조다.

### 네트워크/스트림 지연 측정

별도 서버를 띄울 필요 없이 평소처럼 live pipeline을 실행하면 `/latency`도 같은
8080 포트에서 제공된다. 정확한 측정을 위해 실제 카메라와 모델이 동작하는 상태로 둔다.

```bash
./run_sam3_local.sh   # 또는 ./docker_local.sh
# 이미 필요한 Python/CUDA 환경 안에 있다면: python3 main.py --backend sam3
```

로봇 PC에서는 `gp8_control/tools/measure_perception_latency.py`를 실행한다. 이 도구는
NTP 방식으로 두 PC의 시계 오프셋과 RTT를 먼저 추정하고, 실제 검출 스트림의
`timestamp`와 `elapsed_s`를 비교해 `GP8_PERCEPTION_LATENCY_S` 권장값을 출력한다.

### 검출 레코드 스키마 (jsonl / `/detections` / 스트림 공통)
매 프레임 한 줄/한 객체(JSON). 평행 배열로 정렬 일치:
- `schema_version` — 스키마 버전(정수). 포맷 변경 시 증가 → 소비자가 불일치 감지 (`streaming.SCHEMA_VERSION`)
- `timestamp` (epoch s), `elapsed_s` (추론 시간)
- `capture_timestamp` — RealSense global-time 프레임 시각(epoch s), 사용할 수 없으면 `null`
- `capture_age_s` — 실제 프레임 시각부터 추론 시작까지의 시간(초). 센서/USB/align/crop 포함
- `capture_timestamp_domain` — RealSense timestamp domain 진단 문자열
- `server_send_timestamp` — 스트림 응답을 직렬화·전송하기 직전의 서버 epoch 시각
  (각 클라이언트의 `/detections/stream` 레코드에만 추가)
- `bounding_boxes` — 객체별 네 모서리 `[[TL],[TR],[BR],[BL]]`; 각 점은
  belt-plane `[X,Y,Z]` 좌표(m)
- `class_names` — `["metal"|"transparent"|"cardboard", ...]`
- `confidences` — `[float, ...]` 객체별 top score

### 결과 파일
- `results_local/detections.jsonl` — 위 스키마로 매 프레임 한 줄 누적 기록
- `tmp_results/results_0.2_0.4/{counter}.jpg` — 매 프레임 annotated 이미지 (기존 `object_detector` 동작 그대로)

### 트러블슈팅
| 증상 | 확인 |
|---|---|
| `permission denied ... docker daemon` | docker 그룹 적용 안 됨 → `newgrp docker` 또는 재로그인. 임시: `sg docker -c ./docker_local.sh` |
| `RuntimeError: No device connected` | 카메라 USB 재연결, USB 3.x 포트 사용 확인 (`lsusb` USB 2.x로 잡히면 대역폭 부족) |
| 클라이언트 접속 안 됨 | 방화벽: `sudo ufw status` 확인 후 필요 시 `sudo ufw allow 8080/tcp` |
| 영상은 나오는데 박스 없음 | 객체가 프롬프트(`metal`/`transparent`/`cardboard`) 범주 밖이거나 클래스별 최종 컷 미만 — 카메라 화각/조명 조정, `sam3_detector.default_args()`의 `*_min_score` 확인 |
| FPS 낮음 | 콘솔의 `objs in 0.XXs` 확인. RTX 4090에서 SAM3 ~0.07s(14 FPS), DINO ~80ms(12 FPS)가 정상 |
| `SAM3 runtime files missing` | engine 미빌드 → `bash download_sam3_onnx.sh runtime && bash wrap_sam3_trt.sh` |
| SAM3인데 `RuntimeError: No device connected` | `run_sam3_local.sh` 대신 호스트에서 직접 `main.py`를 띄운 경우. USB 노드가 root 소유라 컨테이너 경유가 필요하다 |
| `tensorrt_cu13` / CUDA 13 관련 로드 실패 | 드라이버가 CUDA 12.x인데 cu13 빌드가 깔린 것 → `pip install "tensorrt-cu12<11"` |

### 포트/해상도 변경
`main.py` 상단 상수:
```python
COLOR_W, COLOR_H, FPS = 640, 480, 30
STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 80   # 50~95, 낮을수록 대역폭↓ 화질↓
```

---

## Perception evaluation (capture + per-class scores)

검출 모듈을 평가하려고 카메라 이미지를 모으고, 박스마다 세 클래스
(`transparent` / `metal` / `cardboard`) 점수를 모두 CSV로 뽑는 2단계 워크플로우.

관련 스크립트는 모두 `scripts/` 하위 폴더에 모아둠.

### Step 1 — 카메라 캡처 (raw JPG만 저장)
```bash
cd /PublicSSD/iitp
./scripts/docker_capture.sh -i 1.0
# 기본 출력: perception_tests/results/perception_eval_<오늘날짜YYMMDD>/images/000000.jpg, 000001.jpg, ...
#   (예: 2026-05-22 → perception_tests/results/perception_eval_260522/)
# 옵션: -i <초> 간격, -o <출력 폴더>
# Ctrl+C 로 정상 종료
```
`docker_capture.sh` 는 `iitp_local:latest` 이미지를 USB 패스스루로 띄우고
`scripts/capture_only.py` 만 실행함 (검출/스트리밍 없음). 어디서
실행하든 자동으로 프로젝트 루트로 cd 하므로 경로 신경 안 써도 됨.

### Step 2 — 평가 실행
```bash
docker run -it --rm --gpus all --ipc=host -v $PWD:/mnt \
  --name iitp_eval chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python scripts/eval_detector.py -i perception_tests/results/perception_eval_260522"
```
`-i` 기본값은 오늘 날짜 폴더 (`perception_tests/results/perception_eval_<YYMMDD>`) 이므로
같은 날 캡처/평가하면 생략 가능. 출력:
- `perception_tests/results/perception_eval_260522/annotated/<원본이름>.jpg` — 박스가 그려진 이미지
- `perception_tests/results/perception_eval_260522/scores.csv` — NMS 통과한 박스마다 한 줄,
  컬럼: `image, box_id, x1, y1, x2, y2, score_transparent, score_metal,
  score_cardboard, top_score, predicted_class`

`score_*` 는 Grounding-DINO 가 각 클래스 토큰 스팬에 대해 낸 sigmoid 점수의
최댓값이라 한 박스 안에서 세 점수를 직접 비교 가능 (헷갈리는 케이스 진단용).

---

## Tests

`perception_tests/` 에 stdlib `unittest` 기반 테스트가 있다. 별도 설치 없이 실행:

```bash
# 호스트: 순수 로직(grouping/tracking/analyze/corrections) 테스트가 돌고,
# 엔진/eval 테스트는 torch가 없어 자동 skip 된다.
python3 -m unittest discover -s perception_tests

# 전체(엔진 NMS/IoU/기하 + eval 클래스별 점수 포함): grounded_sam 컨테이너에서
docker run -i --rm --gpus all --ipc=host -v $PWD:/mnt \
  --name iitp_test chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python -m unittest discover -s perception_tests"
```

| 테스트 파일 | 대상 | 실행 환경 |
|---|---|---|
| `test_group_by_object.py` | `iou`, IoU 클러스터링 | 호스트 |
| `test_group_by_track.py` | `is_ahead`/`centroid`/`distance`, 트랙 생성 | 호스트 (cv2) |
| `test_analyze_tracks.py` | 혼동행렬·정확도 집계 | 호스트 |
| `test_apply_corrections.py` | delete/reassign 보정 적용 | 호스트 (cv2) |
| `test_streaming.py` | `build_record`, /detections·/detections/stream 엔드포인트 | 호스트 |
| `test_sam3_detector.py` | SAM3 인스턴스 → 검출 계약 변환, 라벨/정렬/클래스 id | 호스트 (cv2+pillow) |
| `test_fake_stream.py` | 더미 producer `random_detections` 정합성 | 호스트 |
| `test_engine.py` | `_iou_xyxy`, `nms_by_label`, 기하/마스크 | 컨테이너 (torch) |
| `test_eval_detector.py` | `class_spans`, `per_class_scores` | 컨테이너 (torch) |

카메라/모델이 필요한 경로(`main.py` live, `eval_detector.main`)는 단위 테스트 대상이 아니라
실제 실행(스모크)으로 확인한다. 새 기능은 먼저 `perception_tests/`에 케이스를 추가(red)하고 구현(green)하는 식으로 관리.
