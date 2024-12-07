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

    # Open the text file to write the output
    with open('./data/data_matrices.txt', 'w') as txt_file:
        for outl in output_reader:
            line_count += 1
            if outl[-1] == '1':  # Assuming the last element is '1' for skipping
                next(input_reader)
                continue
            inl = next(input_reader)

            # Ensure the video and frame numbers match between input and output
            assert outl[0] == inl[0]
            assert outl[1] == inl[1]

            # Collect outputs
            outputs.append((float(outl[2]), float(outl[3]), float(outl[4]), float(outl[5])))
            try:
                inputs.append((float(inl[2]), float(inl[3])))
            except ValueError:
                inputs.append((float(inl[2][1]), float(inl[3][1])))

            # Write the formatted output to the text file (in the form of the lists)
            txt_file.write(f"{outl}\n")
            txt_file.write(f"{inl}\n")
        
        print(f'Processed {line_count} lines.')

    return np.array(inputs), np.array(outputs)


if __name__ == "__main__":
    inputs, outputs = get_data()
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
