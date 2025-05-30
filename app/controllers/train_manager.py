import multiprocessing

training_processes = {}

def start_training(app, model_type, epochs, batch_size, learning_rate, user_id, train_func):
    if user_id in training_processes:
        raise RuntimeError("Training already in progress for this user")
    p = multiprocessing.Process(target=train_func, args=(app, model_type, epochs, batch_size, learning_rate, user_id))
    p.start()
    training_processes[user_id] = p

def stop_training(user_id):
    p = training_processes.get(user_id)
    if p and p.is_alive():
        p.terminate()  # 强制杀死进程
        p.join()
        del training_processes[user_id]
        return True
    return False

def is_training(user_id):
    p = training_processes.get(user_id)
    return p.is_alive() if p else False
