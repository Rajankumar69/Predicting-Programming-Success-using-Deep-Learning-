import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocessing import preprocess_data
from utils.visualization import plot_training_history
from config import DATA_PATH, MODEL_PATH, INPUT_SHAPE, EPOCHS, BATCH_SIZE, RANDOM_STATE


# Generate synthetic data (replace this with real data)
def generate_data(save_path):
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    data = {
        'budget': np.random.randint(50, 200, n_samples),
        'team_size': np.random.randint(3, 10, n_samples),
        'duration': np.random.randint(6, 24, n_samples),
        'risk_level': np.random.choice(['low', 'medium', 'high'], n_samples),
        'success': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def main():
    # Generate or load data
    if not os.path.exists(DATA_PATH):
        print("Generating synthetic data...")
        df = generate_data(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    # Preprocess
    df, le, scaler = preprocess_data(df)
    X = df.drop('success', axis=1)
    y = df['success']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(INPUT_SHAPE,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    # Evaluate and save
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    model.save(MODEL_PATH)
    plot_training_history(history)


if __name__ == "__main__":
    main()