import os

def obtain_data():
    # Load dirs name
    cur_dir = os.path.realpath('.')
    data_dir = os.path.join(cur_dir,'Dataset')
    # Obtaining directories
    train_dir = os.path.join(data_dir,'Training')
    print(train_dir)
    test_dir = os.path.join(data_dir,'Testing')
    print(test_dir)

    list_train = []
    list_test = []
    # Obtain train files
    for x in os.listdir(train_dir):
        if(x.endswith(".csv")):
            list_train.append(os.path.join(train_dir, x))

    for x in os.listdir(test_dir):
        if(x.endswith(".csv")):
            list_test.append(os.path.join(test_dir, x))


    # Sorting array
    list_train = sorted(list_train)
    list_test = sorted(list_test)

    print(list_train)
    print(list_test)

    return list_train, list_test
    
obtain_data()
