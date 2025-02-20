def initialize_model(eeg_input_shape, nirs_input_shape):
    # EEG input
    eeg_input = Input(shape=eeg_input_shape, name='eeg_input')
    eeg_conv1 = Conv2D(32, (3, 3), activation='relu')(eeg_input)
    eeg_pool1 = MaxPooling2D((2, 2))(eeg_conv1)
    eeg_conv2 = Conv2D(64, (3, 3), activation='relu')(eeg_pool1)
    eeg_pool2 = MaxPooling2D((2, 2))(eeg_conv2)
    eeg_reshape = Reshape((-1, 64))(eeg_pool2)  # Reshape for LSTM
    eeg_lstm = LSTM(64, return_sequences=True)(eeg_reshape)  # Add LSTM layer
    eeg_flatten = Flatten()(eeg_lstm)

    # NIRS input
    nirs_input = Input(shape=nirs_input_shape, name='nirs_input')
    nirs_conv1 = Conv2D(32, (3, 3), activation='relu')(nirs_input)
    nirs_pool1 = MaxPooling2D((2, 2))(nirs_conv1)
    nirs_conv2 = Conv2D(64, (3, 3), activation='relu')(nirs_pool1)
    nirs_pool2 = MaxPooling2D((2, 2))(nirs_conv2)
    nirs_reshape = Reshape((-1, 64))(nirs_pool2)  # Reshape for LSTM
    nirs_lstm = LSTM(64, return_sequences=True)(nirs_reshape)  # Add LSTM layer
    nirs_flatten = Flatten()(nirs_lstm)

    # Concatenate the EEG and NIRS features
    concatenated = concatenate([eeg_flatten, nirs_flatten])

    # Dense layers
    dense1 = Dense(64, activation='relu')(concatenated)
    dense2 = Dense(64, activation='relu')(dense1)

    # Output layer
    output = Dense(num_classes, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=[eeg_input, nirs_input], outputs=output)

    return model


def inner_loop(model, eeg_task_params, nirs_task_params, X_task, y_task, inner_optimizer, inner_steps=5):

    model.set_weights(initialized_model.get_weights())

    for _ in range(inner_steps):
        with tf.GradientTape() as tape:
            eeg_data_task, nirs_data_task = X_task

            # Forward pass
            y_pred = model([eeg_data_task, nirs_data_task])
            task_loss = tf.keras.losses.categorical_crossentropy(y_task, y_pred)

        gradients = tape.gradient(task_loss, model.trainable_variables)
        inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model


def outer_loop(model, eeg_task_params, nirs_task_params, X_task, y_task, outer_optimizer):


    with tf.GradientTape() as tape:
        # Extract EEG and NIRS data from X_task
        eeg_data_task, nirs_data_task = X_task

        # Forward pass
        y_pred = model([eeg_data_task, nirs_data_task])
        task_loss = tf.keras.losses.categorical_crossentropy(y_task, y_pred)

    gradients = tape.gradient(task_loss, model.trainable_variables)
    outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model



def sample_task(task_distribution):

    task = np.random.choice(task_distribution)
    eeg_params = task["eeg_params"]
    nirs_params = task["nirs_params"]
    return {"eeg_params": eeg_params, "nirs_params": nirs_params}