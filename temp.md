# drone-image-segmentation

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd drone_segmentation
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data:**
    - This project assumes the Semantic Drone Dataset is available at the paths specified in `src/config.py` (defaults to Kaggle input paths).
    - If running locally, download the dataset and update `IMAGE_DIR`, `MASK_DIR`, and `CSV_PATH` in `src/config.py`.

## Usage

### Training

Modify hyperparameters or paths in `src/config.py` as needed. Then run the training script:

```bash
python scripts/train.py


```
