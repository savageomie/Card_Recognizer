import os
import sys
import numpy as np
from typing import Tuple, Dict

# Suppress TensorFlow info and oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.metrics import AUC
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install TensorFlow with: pip install tensorflow")
    sys.exit(1)

try:
    from utils.preprocessing import preprocess_image
except ImportError as e:
    print(f"Error importing preprocessing: {e}")
    print("Ensure utils/preprocessing.py exists in your project")
    sys.exit(1)

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# Rank mapping with validation
RANK_MAP = {
    'A': 12, '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7, '0': 8,  # 0 represents 10
    'J': 9, 'Q': 10, 'K': 11
}

def build_model() -> Model:
    """Builds the multi-output CNN model"""
    try:
        inputs = Input(shape=(*IMG_SIZE, 3))
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Flatten()(x)

        suit = Dense(4, activation='softmax', name='suit')(Dense(64, activation='relu')(x))
        rank = Dense(13, activation='softmax', name='rank')(Dense(64, activation='relu')(x))
        damage = Dense(1, activation='sigmoid', name='damage')(Dense(64, activation='relu')(x))

        return Model(inputs=inputs, outputs=[suit, rank, damage])
    except Exception as e:
        print(f"Error building model: {e}")
        sys.exit(1)

def load_data(folder: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads and preprocesses training data"""
    suits = ['clubs', 'diamonds', 'hearts', 'spades']
    images, suit_labels, rank_labels, damage_labels = [], [], [], []

    try:
        for suit_idx, suit in enumerate(suits):
            suit_dir = os.path.join(folder, suit)
            if not os.path.exists(suit_dir):
                print(f"Warning: Missing directory {suit_dir}")
                continue
                
            for img_file in os.listdir(suit_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                try:
                    # Extract rank code (e.g., "SA" from "cards-[SA]-001.jpg")
                    code_part = img_file.split('[')[1].split(']')[0].upper()
                    rank_char = code_part[1] if len(code_part) > 1 else '0'
                    rank = RANK_MAP.get(rank_char, 0)
                    
                    img_path = os.path.join(suit_dir, img_file)
                    img = preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        suit_labels.append(suit_idx)
                        rank_labels.append(rank)
                        damage_labels.append(0)
                except (IndexError, KeyError) as e:
                    print(f"Warning: Invalid filename format {img_file}: {e}")
                    continue
    
        # Load defaulted cards
        defaulted_dir = os.path.join(folder, 'defaulted')
        if os.path.exists(defaulted_dir):
            for img_file in os.listdir(defaulted_dir):
                try:
                    img_path = os.path.join(defaulted_dir, img_file)
                    img = preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        suit_labels.append(0)
                        rank_labels.append(0)
                        damage_labels.append(1)
                except Exception as e:
                    print(f"Error processing defaulted image {img_file}: {e}")
        
        if len(images) == 0:
            raise ValueError("No valid training images found")
            
        return np.array(images), {
            'suit': to_categorical(suit_labels, 4),
            'rank': to_categorical(rank_labels, 13),
            'damage': np.array(damage_labels)
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def main():
    try:
        # Create models directory if not exists
        os.makedirs('models', exist_ok=True)
        
        model = build_model()
        model.compile(
            optimizer='adam',
            loss={
                'suit': 'categorical_crossentropy',
                'rank': 'categorical_crossentropy',
                'damage': 'binary_crossentropy'
            },
            metrics={
                'suit': ['accuracy'],
                'rank': ['accuracy'],
                'damage': ['accuracy', AUC(name='auc')]
            }
        )
        
        print("Loading training data...")
        X_train, y_train = load_data('data/train')
        X_val, y_val = load_data('data/val')
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )
        
        model.save('models/card_model.h5')
        print("Model saved successfully")
        
        # Print final metrics
        print("\nTraining Results:")
        print(f"Suit Accuracy: {history.history['suit_accuracy'][-1]:.2%}")
        print(f"Rank Accuracy: {history.history['rank_accuracy'][-1]:.2%}")
        print(f"Damage Accuracy: {history.history['damage_accuracy'][-1]:.2%}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()