import random  # Library for generating random numbers

random.seed(0)  # Set seed for reproducibility

def read_csv(filename):
    """
    Purpose: Read data from a CSV file and extract relevant information.

    Description of Input Parameters:
    - filename: Name of the CSV file to be read.

    Description of Return Data:
    - data: List of lists containing extracted data from the CSV file.

    Libraries Used: None
    """
    try:
        lines = open(filename, "r").read().split("\n")
        
        data = []
        for line in lines[1:]:
            if line.replace(" ", "") != "":
                words = line.split(",")
                data.append([float(words[1]), float(words[2]), float(words[3]), float(words[4]), words[5][1:-1]])
        
        return data
    
    except Exception as exception:
        print(exception)
        return None


def encode(data, lookup):
    """
    Purpose: Encode categorical data in the dataset using a lookup table.

    Description of Input Parameters:
    - data: List of lists representing the dataset.
    - lookup: Dictionary mapping categorical values to encoded values.

    Description of Return Data: None

    Libraries Used: None
    """
    try:
        for row in data:
            row[-1] = lookup[row[-1]]
    except Exception as exception:
        print(exception)


def head(data, rows=5):
    """
    Purpose: Display the first few rows of the dataset.

    Description of Input Parameters:
    - data: List of lists representing the dataset.
    - rows: Number of rows to display (default is 5).

    Description of Return Data: None

    Libraries Used: None
    """
    for i in range(rows):
        try:
            print(data[i])
        except:
            break


def shuffle(data):
    """
    Purpose: Shuffle the order of rows in the dataset.

    Description of Input Parameters:
    - data: List of lists representing the dataset.

    Description of Return Data: None

    Libraries Used: random
    """
    try:
        random.shuffle(data)
    except Exception as exception:
        print(exception)


def split_data(data, split_ratio):
    """
    Purpose: Split the dataset into training and testing sets.

    Description of Input Parameters:
    - data: List of lists representing the dataset.
    - split_ratio: Proportion of data to be allocated to the training set.

    Description of Return Data:
    - train_set: Subset of data for training.
    - test_set: Subset of data for testing.

    Libraries Used: None
    """
    try:
        train_size = int(len(data) * split_ratio)
        train_set = data[0:train_size]
        test_set = data[train_size:]
        return train_set, test_set
    
    except Exception as exception:
        print(exception)
        return None, None
