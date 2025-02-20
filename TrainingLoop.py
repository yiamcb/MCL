for meta_iteration in range(100):
    # Inner loop: Train the model on the task's dataset
    model_inner = inner_loop(initialized_model, task_params["eeg_params"], task_params["nirs_params"], X_task, y_task, inner_optimizer)

    # Outer loop: Update the meta-parameters based on the computed gradients
    model_outer = outer_loop(model_inner, task_params["eeg_params"], task_params["nirs_params"], X_task, y_task, outer_optimizer)

    print(f"Meta-Iteration: {meta_iteration}")