import os

def move_file_to_jsonl(save_dir, save_path):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_list = os.listdir(save_dir)
    if file_list and not os.path.exists(save_path):
        with open(save_path, 'w'):
            pass

    for file in file_list:
        path = os.path.join(save_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()
            with open(save_path, 'a') as f:
                for line in lines:
                    if line.strip():
                        f.write(line)

        os.remove(path)

