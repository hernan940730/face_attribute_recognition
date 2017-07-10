import sys, getopt, os, cv2
import numpy as np

def load_names(dataset_folder):
    print ("Loading names...")
    f_data = open(os.path.join(dataset_folder, "Eval/list_eval_partition.txt"), "r")
    names_train = []
    names_test = []
    
    lines = f_data.readlines()
    
    count = 0
    
    for line in lines:
        count += 1
        
        splited_line = line.split()
        img_name = splited_line[0]
        opt = int(splited_line[1])
        
        if opt == 0:
            names_train.append(img_name)
        elif opt == 1:
            names_test.append(img_name)
        else:
            break
    
    print ("Names loaded.")
    return (names_train, names_test)

def load_data(img_folder, names, attr, from_i, chunk):
    print ("Loading data...")
    x = []
    y = []
    
    count = 0
    for i in range(from_i, min(len(names), chunk + from_i)):
        img_name = names[i]
        count += 1
        
        img = cv2.imread(os.path.join(img_folder, img_name))
        img = cv2.resize(img, (224, 224))
        x.append(img)
        y.append(attr["images"][img_name])
    
    print ("Data loaded.")
    return (np.array(x), y)
    
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

def load_file(file_path):
    f = open(file_path, "r")
    return f

def load_args(argv):
    '''
    Load the arguments given by the user
    '''
    
    args_array = ["image_path", "train_model", "weights_path", "dataset_path", "session_path", "epochs", "batch_size", "dataset_img", "chunk", "validation"]
    
    args_map = { key : None for key in args_array}
    
    try:
        opts, args = getopt.getopt(argv, "i:t:w:d:s:e:c:", [arg + "=" for arg in args_array])
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
        elif opt in ("-v", "--validation"):
            if arg in ("True", "False"):
                args_map["validation"] = bool(arg)
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
        elif opt in ("-c", "--chunk"):
            args_map["chunk"] = int(arg)
        elif opt in ("--dataset_img"):
            args_map["dataset_img"] = arg
        else:
            print ('Invalid option', argv)
            sys.exit(2)
            
    return args_map
