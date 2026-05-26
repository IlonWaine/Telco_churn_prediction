from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_PATH = Path(PROJECT_ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

DATA_ANALISYS_DATA = Path(PROJECT_ROOT / 'data_analisys'  )