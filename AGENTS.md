# AGENTS.md

## Commands

- **Install deps**: `pip install -r training/requirements.txt -c training/constraints.txt`
- **Run training**: `python -m training.<module>.train_models` (runs on RunPod GPU, not locally)
- **Type check**: `pyright` (basic mode via pyrightconfig.json)
- **Run single test**: `python week2/test_full_search.py` (standalone scripts, no pytest)

## Architecture

- `training/` - Training scripts per project (hn_predict, msmarco_search, image_caption, etc.)
- `models/` - PyTorch model definitions (CNN, image_caption, hn_predict, msmarco_search)
- `common/` - Shared utilities: `arguments.py` (argparse), `utils.py` (device, logging)
- `api/`, `ui/` - Flask API and Streamlit UI for inference
- `data/` - Local data files (.pt, .pth, .parquet)

## Code Style

- Python 3.x with PyTorch, type hints optional (basic pyright)
- Imports: stdlib → torch → third-party → local (`from common import arguments, utils`)
- Run modules with `python -m training.module.script` pattern
- Use `common.arguments.get_parser()` for CLI args; `common.utils` for device selection
- Models in `models/`, training logic in `training/<project>/`
- Use wandb for experiment tracking with hyperparameters dict
