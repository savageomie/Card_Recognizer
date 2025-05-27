import os
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_image

# Inverse rank mapping for prediction
RANK_MAP_INV = {
    0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7',
    6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q',
    11: 'K', 12: 'A'
}

def predict_card(image_path, model):
    img = preprocess_image(image_path)
    if img is None:
        return "defaulted", None, True
    
    img_array = tf.expand_dims(img, 0) / 255.0
    suit_pred, rank_pred, damage_pred = model.predict(img_array)
    
    suits = ['clubs', 'diamonds', 'hearts', 'spades']
    suit = suits[np.argmax(suit_pred)]
    rank = RANK_MAP_INV[np.argmax(rank_pred)]
    damaged = damage_pred[0][0] > 0.5
    
    # Override to defaulted if confidence is low
    if max(suit_pred[0]) < 0.5 or max(rank_pred[0]) < 0.5:
        return "defaulted", None, True
    
    return suit, rank, damaged

if __name__ == '__main__':
    model = tf.keras.models.load_model('models/card_model.h5')
    image_path = input("Enter image path: ")
    suit, rank, damaged = predict_card(image_path, model)
    
    if suit == "defaulted":
        print("Card: DEFAULTED (invalid/damaged)")
    else:
        print(f"Card: {rank} of {suit} | Damaged: {'Yes' if damaged else 'No'}")