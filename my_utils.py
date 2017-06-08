import sys, getopt, os, cv2
import numpy as np

def load_data(dataset_folder, attr):
    print ("Loading data...")
    f_data = open(os.path.join(dataset_folder, "Eval/list_eval_partition.txt"), "r")
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    img_folder = "/home/hernan940730/Downloads/Inteligentes/Face Attributes Recognizer/img_align_celeba"
    lines = f_data.readlines()
    
    count = 0
    for line in lines:
        count += 1
        
        if (count % 1000 == 0 ):
            print ("Loading:", count / len(lines) * 100, "%")
        
        splited_line = line.split()
        img_name = splited_line[0]
        opt = int(splited_line[1])
        img = cv2.imread(os.path.join(img_folder, img_name))
        img = cv2.resize(img, (224, 224))
        
        if opt == 0 and count <= 50:
            x_train.append(img)
            y_train.append(attr["images"][img_name])
        elif opt == 1 or (count > 50 and count <= 60):
            x_test.append(img)
            y_test.append(attr["images"][img_name])
        else:
            break
    
    print ("Data loaded.")
    return (np.array(x_train), y_train, np.array(x_test), y_test)
    
def load_attributes(file_path):
    f = open(file_path, "r")
    
    attr = {
        "img_count": 0,
        "labels_count" : 0,
        "labels": [],
        "images": {}
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
        attr["images"][line[0]] = bits
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
    
    args_array = ["image_path", "train_model", "weights_path", "dataset_path", "session_path", "epochs", "batch_size"]
    
    args_map = { key : None for key in args_array}
    
    try:
        opts, args = getopt.getopt(argv, "i:t:w:d:s:e:", [arg + "=" for arg in args_array])
    except getopt.GetoptError:
        print ('Invalid option', argv)
        sys.exit(2)
    for opt, arg in opts:        
        if opt in ("-i", "--image_path"):
            args_map["image_path"] = arg
        elif opt in ("-t", "--train_model"):
            if arg in ("True", "False"):
                args_map["train_model"] = bool(arg)
            else:
                print ('Invalid option', opt, arg)
                sys.exit(2)
        elif opt in ("-w", "--weights_path"):
            args_map["weights_path"] = arg
        elif opt in ("-d", "--dataset_path"):
            args_map["dataset_path"] = arg
        elif opt in ("-s", "--session_path"):
            args_map["session_path"] = arg
        elif opt in ("-e", "--epochs"):
            args_map["epochs"] = int(arg)
        elif opt in ("-b", "--batch_size"):
            args_map["batch_size"] = int(arg)
        else:
            print ('Invalid option', argv)
            sys.exit(2)
            
    return args_map
