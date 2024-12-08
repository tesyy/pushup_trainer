import csv
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(type_data,neighbor):
    if type_data == "train":
        output_vectors = open('./data/output_vectors.csv')
        input_vectors = open('./data/New_input_vectors.csv')
    elif type_data == "test":
        output_vectors = open('./data/test_output_vectors.csv')
        input_vectors = open('./data/test_input_vectors.csv')
    else:
        assert False ("wrong parameter input: type_data ")

    output_reader = csv.reader(output_vectors, delimiter=',')
    input_reader = csv.reader(input_vectors, delimiter=',')
    line_count = 0

    outputs = []
    inputs = []

    # Open the text file to write the output
    with open('./data/data_matrices.txt', 'w') as txt_file:
        for outl,inl in zip(output_reader,input_reader):
            line_count += 1
            #print(outl,inl)
            if outl[-1] == '1':  # Assuming the last element is '1' for skipping
                continue
            # Ensure the video and frame numbers match between input and output
            assert outl[0] == inl[0], f"outl0 is {outl[0]} inl0 is {inl[0]}, Processed {line_count} lines."
            assert outl[1] == inl[1], f"outl1 is {outl[1]} inl1 is {inl[1]}, Processed {line_count} lines."

            # Collect outputs
            outputs.append((float(outl[2]), float(outl[3]), float(outl[4]), float(outl[5])))
            inputs.append([float(inl[2]), float(inl[3]),float(inl[4]),float(inl[5])])
            # Write the formatted output to the text file (in the form of the lists)
            txt_file.write(f"{outl}\n")
            txt_file.write(f"{inl}\n")
        inputLen = len(inputs)
        with open('./data/input_matrices.csv', 'w') as txt_file:
            for i in range(0,inputLen):
                #neighbor = 34
                for k in range(0,4):
                    #if inputs[i][k] == 0:
                    sum = 0
                    right_skipped = 0
                    left_skipped = 0
                    right_count = 0
                    left_count = 0
                    for j in range(0,neighbor):
                        if (i+j+1 < inputLen and right_count < neighbor/2) or left_skipped >= neighbor/2:
                            sum += inputs[i+j+1][k]
                            right_count += 1
                        else:
                            right_skipped += 1
                        if (i-j-1 > 0  and left_count < neighbor/2) or right_skipped >= neighbor/2:
                            sum += inputs[i-j-1][k]
                            left_count += 1
                        else:
                            left_skipped += 1
                    sum = sum/neighbor if neighbor > 0 else 0
                    inputs[i][k] = sum
                txt_file.write(f"{inputs[i]}\n")
                
        #print(f'Processed {line_count} lines.')
    scaler = StandardScaler() 
    inputs = scaler.fit_transform(inputs)
    return np.array(inputs), np.array(outputs)


if __name__ == "__main__":
    inputs, outputs = get_data()
    #print(f"Inputs: {inputs}")
    #print(f"Outputs: {outputs}")
