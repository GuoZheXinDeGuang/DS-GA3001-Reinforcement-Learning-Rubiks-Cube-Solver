import tensorflow as tf
from tensorflow import keras

def create_dense_model(input_dim):
    """
    Build a dense neural network for value and policy outputs.
    """
    state_input = keras.Input(shape=(input_dim,), name='state_input')
    x = keras.layers.Dense(4096, activation='elu')(state_input)
    x = keras.layers.Dense(2048, activation='elu')(x)
    # separate streams for policy and value
    policy_hidden = keras.layers.Dense(512, activation='elu')(x)
    value_hidden  = keras.layers.Dense(512, activation='elu')(x)
    policy_output = keras.layers.Dense(12, activation='softmax', name='policy')(policy_hidden)
    value_output  = keras.layers.Dense(1, name='value')(value_hidden)
    return keras.Model(inputs=state_input, outputs=[value_output, policy_output])

def setup_optimizer(model, lr_rate):
    """
    Compile the model with RMSprop optimizer and appropriate losses.
    """
    optimizer = keras.optimizers.RMSprop(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            'value':  'mean_squared_error',
            'policy': 'sparse_categorical_crossentropy'
        }
    )
    return model
