from keras import layers
from keras import Model
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, ReLU

import tensorflow as tf


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same',
                                   kernel_initializer='he_normal', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2D(planes, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer='he_normal', use_bias=False)
        self.bn2 = BatchNormalization()
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = GlobalAveragePooling2D()
            self.conv3 = Conv2D(planes // 16, kernel_size=1, strides=1,
                                       kernel_initializer='he_normal', use_bias=True)
            self.conv4 = Conv2D(planes, kernel_size=1, strides=1,
                                       kernel_initializer='he_normal', use_bias=True)

    def call(self, inputs):
        residual = inputs
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w)
            w = layers.Activation('sigmoid')(w)

            out = out * w

        out = layers.add([out, residual])
        out = self.relu(out)

        return out

class ResNet(Model):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.bn = layers.BatchNormalization()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(planes * block.expansion, kernel_size=1,
                       strides=stride, use_bias=False),
                BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return Sequential(layers)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = tf.reshape(x, (-1, self.inplanes))
        x = self.bn(x)
        return x

import tensorflow as tf

class VideoCNN(tf.keras.Model):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()

        # frontend3D
        self.frontend3D = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')
                ])

        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.dropout = tf.keras.layers.Dropout(0.5)

        # backend_gru
        # initialize
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        x = self.frontend3D(x)
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        x = tf.reshape(x, [-1, 64, x.shape[3], x.shape[4]])
        x = self.resnet18(x)
        return x

    def call(self, x):
        b, t = x.shape[:2]

        x = self.visual_frontend_forward(x)

        #x = self.dropout(x)
        feat = tf.reshape(x, [b, -1, 512])

        x = tf.reshape(x, [b, -1, 512])
        return x

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, tf.keras.layers.Conv3D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.filters
                m.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
                if m.use_bias:
                    m.bias_initializer = tf.keras.initializers.Zeros()

            elif isinstance(m, tf.keras.layers.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.filters
                m.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
                if m.use_bias:
                    m.bias_initializer = tf.keras.initializers.Zeros()

            elif isinstance(m, tf.keras.layers.Conv1D):
                n = m.kernel_size[0] * m.filters
                m.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
                if m.use_bias:
                    m.bias_initializer = tf.keras.initializers.Zeros()

            elif isinstance(m, tf.keras.layers.BatchNormalization):
                m.gamma_initializer = tf.keras.initializers.Ones()
                m.beta_initializer = tf.keras.initializers.Zeros()

