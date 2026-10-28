from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # ReCon SL files (not redistributed; see README)
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for p in (MODELS, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD_UNITS = 90       # PUCSL domestic block boundary: above 90 kWh/month the unit price and fixed charge jump
WINDOW_MONTHS = 12         # target = median monthly use in the 12 months up to the survey month
MIN_VALID_MONTHS = 6
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}
BOOT = 500
