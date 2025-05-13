import pandas as pd
import tensorflow as tf
from utils.preprocessing import preprocess_data
from config import MODEL_PATH


def predict_success(new_data):
    # Load model and preprocessors (in a real project, save/load the scaler and encoder)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess new data (assume same steps as training)
    df, le, scaler = preprocess_data(new_data.copy())
    return model.predict(df)


if __name__ == "__main__":
    # Example new program
    new_program = pd.DataFrame({
        'budget': [120],
        'team_size': [6],
        'duration': [18],
        'risk_level': ['medium'],
        'success': [0]  # Placeholder (ignored in prediction)
    })

    prediction = predict_success(new_program)
    print(f"Success probability: {prediction[0][0] * 100:.2f}%")