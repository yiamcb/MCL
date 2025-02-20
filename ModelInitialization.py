num_tasks = 10
num_classes = 2  
eeg_input_shape = np.shape(eeg_data)[1], np.shape(eeg_data)[2], 1

nirs_input_shape = np.shape(nirs_data)[1], np.shape(nirs_data)[2], 1

task_distribution = generate_task_distribution(num_tasks, eeg_input_shape, nirs_input_shape)

# Initialize the multi-input model
initialized_model = initialize_model(eeg_input_shape, nirs_input_shape)

# Set up optimizers
inner_optimizer = Adam(learning_rate=0.0001)
outer_optimizer = Adam(learning_rate=0.0001)

# Sample a task for training
task_params = sample_task(task_distribution)