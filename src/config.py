from pathlib import Path

# ── 프로젝트 루트 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── 데이터 경로 ───────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_SUBMISSION = ROOT / "data" / "submissions"

TRAIN_PATH      = DATA_RAW / "train.csv"
TEST_PATH       = DATA_RAW / "test.csv"
LAYOUT_PATH     = DATA_RAW / "layout_info.csv"
SAMPLE_SUB_PATH = DATA_RAW / "sample_submission.csv"
LAYOUT_ENC_PATH = DATA_PROCESSED / "layout_encoding.csv"

# ── 모델 / 설정 경로 ──────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
CONFIGS_DIR     = ROOT / "configs"
LGBM_CONFIG     = CONFIGS_DIR / "lgbm.yaml"

# ── 타깃 컬럼 ────────────────────────────────────────────────────
TARGET_COL      = "avg_delay_minutes_next_30m"

# ── 드롭 컬럼 (학습에 사용하지 않음) ──────────────────────────────
DROP_COLS       = ["ID", "layout_id", "scenario_id", TARGET_COL]


def gpu_available() -> bool:
    """Return True if an NVIDIA GPU is accessible."""
    import subprocess
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
