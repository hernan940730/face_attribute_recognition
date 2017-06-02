import sys, getopt

def load_attributes(file_path):
    f = open(file_path, "r")
    
    attr = {
        "img_count": int( f.readline() ),
        "labels": [],
        "images": {}
        }
    
    attr["labels"] = [label for label in f.readline().split(" ")]
    
    for i in range(atrr["img_count"]):
        pass

def load_file(file_path):
    f = open(file_path, "r")
    return f

def load_args(argv):
    '''
    Load the arguments given by the user
    '''
    
    args_array = ["image_path", "train_model", "weights_path"]
    
    args_map = { key : None for key in args_array}
    
    try:
        opts, args = getopt.getopt(argv, "i:t:w:", [arg + "=" for arg in args_array])
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
        else:
            print ('Invalid option', argv)
            sys.exit(2)
            
    return args_map
