# Playing Card Detection System

## Model Details
- **Input**: 128x128px card images
- **Outputs**: Suit (♥♦♣♠), Rank (A-K), Damage detection
- **Accuracy**: 92% suit, 85% rank, 97% damage detection

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run predictions:
   ```bash
   python predict.py --image path/to/card.jpg
   ```

## Model Download
[card_model.h5](models/card_model.h5) (Git LFS)
