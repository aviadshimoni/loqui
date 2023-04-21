from keras.layers import Input, Conv2D, MaxPooling2D, GaussianNoise, Flatten, Dense, Dropout, GRU, Bidirectional, \
    Concatenate
from keras.models import Model


class VideoModel(tf.keras.Model):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()

        self.args = args

        self.video_cnn = VideoCNN(args.se)
        if args.border:
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = Bidirectional(GRU(1024, return_sequences=True, dropout=0.2), merge_mode='concat')
        self.v_cls = Dense(args.n_class)
        self.dropout = Dropout(dropout)

    def call(self, v, border=None, training=False):
        if training:
            v = GaussianNoise(0.01)(v)
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v, training=training)

        if self.args.border:
            border = tf.expand_dims(border, axis=-1)
            f_v = Concatenate(axis=-1)([f_v, border])
        h = self.gru(f_v, training=training)
        y_v = tf.reduce_mean(self.dropout(h, training=training), axis=1)

        return self.v_cls(y_v)
