from absl import app

def generate_train_val_txt(total_number):
    full_path = 'train_eval.txt'
    file = open(full_path, 'w')
    for num in range(total_number):
        if num % 10 == 0:
            file.write('2')
            file.write('\n')
        else:
            file.write('1')
            file.write('\n')
    file.close()

if __name__ == '__main__':
    generate_train_val_txt(20)


