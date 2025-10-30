import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from transformers import TFAutoModel  # type: ignore

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128

class IndoBERTBiLSTMClassifier(keras.Model):
    def __init__(self, num_classes, lstm_units=128, dense_units=256, dropout=0.3):
        super().__init__()
        self.bert = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=False)
        self.bert.trainable = False
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))
        self.drop1  = layers.Dropout(dropout)
        self.dense  = layers.Dense(dense_units, activation="relu")
        self.drop2  = layers.Dropout(dropout)
        self.out    = layers.Dense(num_classes, activation="softmax")
        _ = self([tf.zeros((1,MAX_LEN),dtype=tf.int32), tf.zeros((1,MAX_LEN),dtype=tf.int32)])

    def call(self, inputs, training=False):
        ids, mask = inputs
        x = self.bert(ids, attention_mask=mask, training=training)[0]
        x = self.bilstm(x, training=training)
        x = self.drop1(x, training=training)
        x = self.dense(x)
        x = self.drop2(x, training=training)
        return self.out(x)
