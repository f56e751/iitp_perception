아래 명령어들을 실행하기 위해서는 docker 권한이 있어야 함. Docker 권한 부여 방법을 모르는 경우 문의 바람. (송지환)

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

## Local capture + MJPEG stream (robot6 직결 모드)

RealSense 카메라를 이 서버(robot6)에 USB로 직결해서, 네트워크 왕복 없이 바로 SAM 추론을 돌리고 바운딩박스가 그려진 영상을 다른 컴퓨터(로봇 PC 등)의 브라우저로 실시간 송출하는 모드.

### 구성 파일
- `Dockerfile.local` — 기존 `chaehyeonsong/grounded_sam` 이미지 위에 `pyrealsense2`만 얹은 파생 이미지 (`iitp_local:latest`)
- `docker_local.sh` — USB 패스스루(`--privileged`, `-v /dev:/dev`) + `--network host`로 컨테이너 실행
- `capture_and_detect.py` — pyrealsense2로 컬러+깊이 캡처 → `object_detector` 호출 → `results_local/detections.jsonl`에 결과 추가, 포트 8080에서 MJPEG 스트림 송출

### 사전 준비 (최초 1회)
```bash
# 1) iitp 계정에 docker 그룹 추가 (sudo 필요)
sudo usermod -aG docker iitp
# 로그아웃/재로그인 또는 newgrp docker

# 2) 파생 이미지 빌드 (베이스 이미지 chaehyeonsong/grounded_sam:latest가 이미 로컬에 있어야 함)
cd /PublicSSD/iitp
docker build -f Dockerfile.local -t iitp_local:latest .

# 3) RealSense 카메라를 robot6 USB 3.x 포트에 연결
lsusb | grep RealSense        # Intel Corp. Intel(R) RealSense(TM) ... 확인
```

### 서버 실행 (robot6)
```bash
cd /PublicSSD/iitp
./docker_local.sh
```
- 모델 로드(~20초) 후 `MJPEG stream ready: http://<this-host>:8080/stream` 출력되면 준비 완료
- 중단: 콘솔에서 `Ctrl+C` (컨테이너는 `--rm`이라 자동 정리)

SSH 끊어도 계속 돌리고 싶을 때:
```bash
cd /PublicSSD/iitp && nohup ./docker_local.sh > stream.log 2>&1 &
# 중단:
docker stop iitp_local
```

### 클라이언트에서 영상 보기 (다른 컴퓨터)
브라우저 또는 VLC에서:
```
http://147.46.175.15:8080/stream
```
- 같은 캠퍼스 LAN이면 위 IP 사용. 안 되면 `147.46.240.59`도 시도.
- 여러 명이 동시에 접속해도 됨.

### 결과 파일
- `results_local/detections.jsonl` — 매 프레임 한 줄. 필드: `timestamp`, `elapsed_s`, `positions`(카메라 좌표계 (X,Y,Z), m), `class_names`(`metal`/`transparent`/`cardboard`)
- `tmp_results/results_0.2_0.4/{counter}.jpg` — 매 프레임 annotated 이미지 (기존 `object_detector` 동작 그대로)

### 트러블슈팅
| 증상 | 확인 |
|---|---|
| `permission denied ... docker daemon` | docker 그룹 적용 안 됨 → `newgrp docker` 또는 재로그인. 임시: `sg docker -c ./docker_local.sh` |
| `RuntimeError: No device connected` | 카메라 USB 재연결, USB 3.x 포트 사용 확인 (`lsusb` USB 2.x로 잡히면 대역폭 부족) |
| 클라이언트 접속 안 됨 | 방화벽: `sudo ufw status` 확인 후 필요 시 `sudo ufw allow 8080/tcp` |
| 영상은 나오는데 박스 없음 | 객체가 프롬프트(`metal`/`transparent`/`cardboard`) 범주 밖이거나 신뢰도 < 0.2 — 카메라 화각/조명 조정 |
| FPS 낮음 | 콘솔의 `objs in 0.XXs` 확인. RTX 4090에서 ~80ms(12 FPS)가 정상 |

### 포트/해상도 변경
`capture_and_detect.py` 상단 상수:
```python
COLOR_W, COLOR_H, FPS = 640, 480, 30
STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 80   # 50~95, 낮을수록 대역폭↓ 화질↓
```

---

## Perception evaluation (capture + per-class scores)

검출 모듈을 평가하려고 카메라 이미지를 모으고, 박스마다 세 클래스
(`transparent` / `metal` / `cardboard`) 점수를 모두 CSV로 뽑는 2단계 워크플로우.

관련 스크립트는 모두 `perception_eval/` 하위 폴더에 모아둠.

### Step 1 — 카메라 캡처 (raw JPG만 저장)
```bash
cd /PublicSSD/iitp
./perception_eval/docker_capture.sh -i 1.0
# 기본 출력: tmp_results/perception_eval_260520/images/000000.jpg, 000001.jpg, ...
# 옵션: -i <초> 간격, -o <출력 폴더>
# Ctrl+C 로 정상 종료
```
`docker_capture.sh` 는 `iitp_local:latest` 이미지를 USB 패스스루로 띄우고
`perception_eval/capture_only.py` 만 실행함 (검출/스트리밍 없음). 어디서
실행하든 자동으로 프로젝트 루트로 cd 하므로 경로 신경 안 써도 됨.

### Step 2 — 평가 실행
```bash
docker run -it --rm --gpus all --ipc=host -v $PWD:/mnt \
  --name iitp_eval chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python perception_eval/eval_detector.py -i tmp_results/perception_eval_260520"
```
출력:
- `tmp_results/perception_eval_260520/annotated/<원본이름>.jpg` — 박스가 그려진 이미지
- `tmp_results/perception_eval_260520/scores.csv` — NMS 통과한 박스마다 한 줄,
  컬럼: `image, box_id, x1, y1, x2, y2, score_transparent, score_metal,
  score_cardboard, top_score, predicted_class`

`score_*` 는 Grounding-DINO 가 각 클래스 토큰 스팬에 대해 낸 sigmoid 점수의
최댓값이라 한 박스 안에서 세 점수를 직접 비교 가능 (헷갈리는 케이스 진단용).