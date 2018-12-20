from keras.models import Sequential
from keras.layers.core import Dense, Activation
def create_model(k=2):
    model = Sequential()  # 建立模型
    model.add(Dense(input_dim=7, units=7*k))
    model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
    model.add(Dense(input_dim=7*k, units=1))
    model.add(Activation('sigmoid'))  # 由于是0-1输出，用sigmoid函数作为激活函数

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # class_mode = 'binary'
    # 编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
    # 另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
    # 求解方法我们指定用adam，还有sgd、rmsprop等可选
    return model