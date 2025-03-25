import numpy as np
import pandas as pd
from keras.src.layers.normalization.layer_normalization import LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from util import load_data
import matplotlib.pyplot as plt



class SklearnF1Callback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(self.y_val, y_pred_classes, average='weighted')
        logs['val_f1'] = f1
        print(f" — val_f1: {f1:.4f}")

class EarlyStoppingOverfitting(Callback):
    def __init__(self, max_gap=0.2, patience=3):
        super().__init__()
        self.max_gap = max_gap
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if train_loss is not None and val_loss is not None:
            gap = val_loss - train_loss

            if gap > self.max_gap:
                self.wait += 1
                print(f"\nWarning: Overfitting detected (Gap: {gap:.4f}). Patience left: {self.patience - self.wait}")

                if self.wait >= self.patience:
                    print("\nStopping training due to excessive overfitting.")
                    self.model.stop_training = True
            else:
                self.wait = 0 # reset patience if the gap is within limits


x_train, x_test, y_train, y_test = load_data(test_size=0.1)

x_train_train, x_train_validate, y_train_train, y_train_validate = train_test_split(x_train, y_train, test_size=0.16)

y_train_train_cat = to_categorical(y_train_train, num_classes=5)
y_train_validate_cat = to_categorical(y_train_validate, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    Dense(
        218,
        input_dim=x_train_train.shape[1],
        kernel_regularizer=l2(0.001)
    ),
    LayerNormalization(),
    ReLU(),
    Dropout(0.33),

    Dense(128, kernel_regularizer=l2(0.0008)),
    LayerNormalization(),
    ReLU(),
    Dropout(0.16),

    Dense(64),
    LayerNormalization(),
    ReLU(),

    Dense(5, activation='softmax')
])


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.97,
    staircase=True
)

model.compile(
    optimizer=Adam(
        learning_rate=lr_schedule
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'mae']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min'
)

overfitting_stopping = EarlyStoppingOverfitting(max_gap=0.035, patience=4)
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)
history = model.fit(
    x_train_train,
    y_train_train_cat,
    epochs=90,
    batch_size=16,
    validation_data=(x_train_validate, y_train_validate_cat),
    callbacks=[
        SklearnF1Callback(x_train_validate, y_train_validate),
        early_stopping,
        tensorboard_callback,
        overfitting_stopping
    ]
)

model.save('newmodel1.keras')

y_test_pred = model.predict(x_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

final_f1_score = f1_score(y_test, y_test_pred_classes, average='weighted')
print("Final F1 score (weighted)", final_f1_score)

epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, train_mae, label="Training MAE")
plt.plot(epochs, val_mae, label="Validation MAE")
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('MAE over Epochs')
plt.legend()

plt.show()

df_test = pd.read_csv('teste.csv')
x_test_csv = df_test.drop('id', axis=1)

scaler = StandardScaler()
x_test_csv_scaled = scaler.fit_transform(x_test_csv)

pred_y_test_csv = model.predict(x_test_csv_scaled)
pred_y_classes_test_csv = np.argmax(pred_y_test_csv, axis=1)

plt.figure(figsize=(8, 6))
plt.hist(pred_y_classes_test_csv, bins=np.arange(6)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['Planeta Deserto', 'Planeta Vulcânico', 'Planeta Oceânico', 'Planeta Florestal', 'Planeta Gelado'])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Test.csv Classes')
plt.show()

pred_y_test_csv = model.predict(x_test_csv)
pred_y_classes_test_csv = np.argmax(pred_y_test_csv, axis=1)

plt.figure(figsize=(8, 6))
plt.hist(y_train, bins=np.arange(6)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['Planeta Deserto', 'Planeta Vulcânico', 'Planeta Oceânico', 'Planeta Florestal', 'Planeta Gelado'])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Train Classes')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(np.argmax(model.predict(x_test), axis=1), bins=np.arange(6)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['Planeta Deserto', 'Planeta Vulcânico', 'Planeta Oceânico', 'Planeta Florestal', 'Planeta Gelado'])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Test Classes')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(np.argmax(model.predict(x_train_validate), axis=1), bins=np.arange(6)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['Planeta Deserto', 'Planeta Vulcânico', 'Planeta Oceânico', 'Planeta Florestal', 'Planeta Gelado'])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Validation Classes')
plt.show()
