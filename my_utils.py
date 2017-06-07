import sys, getopt, os

def load_data(dataset_folder):
    f_data = open(os.path.join(dataset_folder, "Eval/list_eval_partition.txt"), "r")
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    
    return (x_train, y_train, x_test, y_test)
    
def load_attributes(file_path):
    f = open(file_path, "r")
    
    attr = {
        "img_count": 0,
        "labels_count" : 0,
        "labels": [],
        "img_names": [],
        "attributes" : []
        }
    
    attr["img_count"] = int( f.readline() )
    attr["labels"] = [label for label in f.readline().split()]
    attr["labels_count"] = len(attr["labels"])
    
    attributes = []
    
    for i in range(attr["img_count"]):
        line = f.readline().strip().split()
        bits = []
        for i in range( 1, len(line) ):
            bits.append(1 if int(line[i]) == 1 else 0 )
        attr["img_names"].append( line[0] )
        attributes.append(bits)
    attr["attributes"] = attributes
    return attr

def load_dataset(path):
    pass

def load_file(file_path):
    f = open(file_path, "r")
    return f

def load_args(argv):
    '''
    Load the arguments given by the user
    '''
    
    args_array = ["image_path", "train_model", "weights_path", "dataset_path"]
    
    args_map = { key : None for key in args_array}
    
    try:
        opts, args = getopt.getopt(argv, "i:t:w:d:", [arg + "=" for arg in args_array])
    except getopt.GetoptError:
        print ('Invalid option', argv)
        sys.exit(2)
    for opt, arg in opts:        
        if opt in ("-i", "--image_path"):
            args_map["image_path"] = arg
        elif opt in ("-t", "--train_model"):
            if arg in (True, False):
                args_map["train_model"] = arg
            else:
                print ('Invalid option', opt, arg)
                sys.exit(2)
        elif opt in ("-w", "--weights_path"):
            args_map["weights_path"] = arg
        elif opt in ("-d", "--dataset_path"):
            args_map["dataset_path"] = arg
        else:
            print ('Invalid option', argv)
            sys.exit(2)
            
    return args_map
