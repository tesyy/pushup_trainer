import csv
import numpy as np


def get_data():
    output_vectors = open('./data/output_vectors.csv')
    input_vectors = open('./data/input_vectors.csv')

    output_reader = csv.reader(output_vectors, delimiter=',')
    input_reader = csv.reader(input_vectors, delimiter=',')
    line_count = 0

    outputs = []
    inputs = []

    for outl in output_reader:
        line_count += 1
        if outl[-1] == 1:
            next(input_reader)
            continue
        inl = next(input_reader)

        # print("#########@#@#@")
        # print(outl)
        # print(inl)
        assert outl[0] == inl[0]
        assert outl[1] == inl[1]

        outputs.append((float(outl[2]), float(outl[3]), float(outl[4]), float(outl[5]), float(outl[6])))

        inputs_list = []
        for inp in inl[2:-1]:
            inputs_list.append(float(inp))

        inputs.append(np.array(inputs_list))

    print(f'Processed {line_count} lines.')

    return np.array(inputs), np.array(outputs)


if __name__ == '__main__':
    print(get_data())