# RDE 데이터셋 가이드

다음 세대 에이전트가 학습·평가 코드를 수정하거나 실행할 때 참고할 데이터셋 관련 정보.

Use `uv run python` to execute Python code.

## Dataset location

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

## 데이터 루트

| 항목 | 값 |
|------|-----|
| 기본 경로 | `/data/jayn2u/lab_datasets` (또는 `/mnt/data/lab_datasets`) |
| 환경변수 | `RDE_DATA_ROOT` (설정 시 `--root_dir` 기본값을 덮어씀) |
| CLI 인자 | `--root_dir` (`2024-CVPR-RDE/utils/options.py`) |
| `run_rde.sh` | `root_dir="${RDE_DATA_ROOT:-/data/jayn2u/lab_datasets}"` |

코드는 `{root_dir}/{dataset_dir}/` 아래에서 어노테이션과 이미지를 읽는다.  
실행 cwd는 `2024-CVPR-RDE/` 여야 한다 (`run_rde.sh`, `train.py`, `test.py`).

## 지원 데이터셋

`--dataset_name` 값과 로더 매핑 (`2024-CVPR-RDE/datasets/build.py`):

| `--dataset_name` | 로더 클래스 | 디렉터리명 |
|------------------|------------|-----------|
| `CUHK-PEDES` | `CUHKPEDES` | `CUHK-PEDES` |
| `ICFG-PEDES` | `ICFGPEDES` | `ICFG-PEDES` |
| `RSTPReid` | `RSTPReid` | `RSTPReid` |

### CUHK-PEDES

```
/data/jayn2u/lab_datasets/CUHK-PEDES/
├── imgs/                  # 이미지 (예: imgs/CUHK01/0363004.png)
└── reid_raw.json          # RDE가 사용하는 어노테이션 (필수)
```

- 어노테이션 키: `split`, `captions`, `file_path`, `id`
- 학습 시 PID: `int(anno['id']) - 1` (0부터 시작하도록 보정)
- 평가 시 PID: `int(anno['id'])` (원본 ID 유지)
- split 값: `train` / `test` / `val` (그 외는 val로 분류)

**RDE가 사용하지 않는 파일** (같은 디렉터리에 존재):
- `reid_raw_diverse_color.json`
- `reid_raw_negative_gemma4:26b.json`

### ICFG-PEDES

```
/data/jayn2u/lab_datasets/ICFG-PEDES/
├── imgs/                  # 이미지 (예: imgs/test/0627/....jpg)
└── ICFG-PEDES.json        # RDE가 사용하는 어노테이션 (필수)
```

- 어노테이션 키: `split`, `file_path`, `id`, `captions` (및 `processed_tokens`)
- 학습·평가 모두 PID: `int(anno['id'])` (0부터 연속)
- split 값: `train` / `test` / `val`

**RDE가 사용하지 않는 파일**:
- `captions.csv`, `captions_cleaned.csv`, `invalid_paths.csv`

### RSTPReid

```
/data/jayn2u/lab_datasets/RSTPReid/
├── imgs/                  # 이미지 (예: imgs/0000_c14_0031.jpg)
└── data_captions.json     # RDE가 사용하는 어노테이션 (필수)
```

- 어노테이션 키: `id`, `img_path`, `captions`, `split` (`file_path` 아님)
- 학습·평가 모두 PID: `int(anno['id'])`
- split 값: `train` / `test` / `val`

**RDE가 사용하지 않는 파일**:
- `data_captions_diverse_color.json`
- `data_captions_negative_gemma4:26b.json`

## 이미지 경로 규칙

모든 데이터셋 공통: `img_path = join({root_dir}/{DatasetName}/imgs/, anno 내 상대경로)`

| 데이터셋 | 어노테이션 이미지 필드 | 예시 |
|---------|----------------------|------|
| CUHK-PEDES | `file_path` | `CUHK01/0363004.png` |
| ICFG-PEDES | `file_path` | `test/0627/0627_010_05_0303afternoon_1591_0.jpg` |
| RSTPReid | `img_path` | `0000_c14_0031.jpg` |

로더는 파일 존재 여부를 `_check_before_run()`에서 디렉터리·어노테이션 수준만 검사한다. 개별 이미지 누락 시 `read_image()`에서 IOError가 난다.

## 학습 데이터 형식 (내부)

`ImageTextDataset`에 전달되는 train 튜플: `(pid, image_id, img_path, caption)`

- `image_id`: 어노테이션 내 이미지 단위 순번 (caption마다 동일 image_id 공유)
- noisy injection 후에도 PID·image_id·img_path는 유지되고 caption만 교체될 수 있음

## Noisy correspondence

| 항목 | 설명 |
|------|------|
| 인자 | `--noisy_rate` (0.0 / 0.2 / 0.5 / 0.8), `--noisy_file` |
| 인덱스 위치 | `2024-CVPR-RDE/noiseindex/{DATASET_NAME}_{noisy_rate}.npy` |
| 제공 파일 | 각 데이터셋별 `0.2`, `0.5`, `0.8` ( `0.0` 파일 없음) |
| `noisy_rate=0.0` | 인덱스 파일 불필요 (`inject_noisy_correspondence`가 shuffle 생략) |
| `noisy_rate>0` | `noisy_file`이 있으면 로드, 없으면 랜덤 생성 후 저장 |

논문과 동일한 노이즈 실험 시 반드시 `noiseindex/`의 `.npy`를 사용할 것.

## 검증·테스트 split

- `--val_dataset` 기본값: `test` (학습 중 validation에 test set 사용)
- `train.py`에 ICFG-PEDES만 val을 쓰도록 하는 주석 코드가 있으나 현재 비활성화
- `test.py` 평가는 항상 `dataset.test` 사용

## 실행 예시

```bash
cd /data/jayn2u/RDE/2024-CVPR-RDE
source /data/jayn2u/RDE/.venv/bin/activate

# 학습 (기본: CUHK-PEDES, noisy_rate=0.0)
sh run_rde.sh

# 데이터셋·경로 변경 예시
DATASET_NAME=ICFG-PEDES RDE_DATA_ROOT=/data/jayn2u/lab_datasets sh run_rde.sh

# 평가
python test.py --checkpoint_dir run_logs/CUHK-PEDES/<run_dir>
python test.py --checkpoint_dir run_logs/CUHK-PEDES/<run_dir> --root_dir /data/jayn2u/lab_datasets
```

## 에이전트 주의사항

1. 어노테이션 파일명을 바꾸지 말 것. 로더 경로가 하드코딩되어 있다 (`cuhkpedes.py`, `icfgpedes.py`, `rstpreid.py`).
2. `lab_datasets`의 `*_diverse_color.json`, `*_negative_*.json`은 이 RDE 코드와 무관하다.
3. 새 데이터셋 추가 시 `datasets/build.py`의 `__factory` 등록과 전용 로더 클래스가 필요하다.
4. CUHK-PEDES만 학습 PID에 `-1` 보정이 있어, 다른 데이터셋과 PID 스킴이 다르다.
5. 학습 출력의 `configs.yaml`에 `root_dir`이 저장되므로, 평가 시 데이터 위치가 바뀌면 `--root_dir`로 덮어쓸 것.
