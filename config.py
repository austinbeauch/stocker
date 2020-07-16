import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(
        '---------------------- {0} ----------------------'.format(notifi_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('---------------------- END ----------------------')


def print_config(config):
    content_list = []
    args = list(vars(config))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(config, arg))]
    print_notification(content_list, 'CONFIG')


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")

main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for the model
model_arg = add_argument_group("Model")
model_arg.add_argument("--hidden_dim", type=int,
                       default=50,
                       help="Number of hidden features")

model_arg.add_argument("--num_layers", type=int,
                       default=4,
                       help="Number of recurrent layers")

model_arg.add_argument("--model_type",
                       type=str,
                       default="LSTM",
                       choices=["LSTM", "GRU", "LSTMForcing"],
                       help="Model type, either LSTM or GRU")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--train", type=str,
                       default="./data/train/",
                       help="Training directory containing stock information CSV files")
train_arg.add_argument("--val", type=str,
                       default="./data/val/",
                       help="Validation directory containing stock information CSV files")
train_arg.add_argument("--test", type=str,
                       default="./data/test/",
                       help="Testing directory containing stock information CSV files")
train_arg.add_argument("--test_file", type=str,
                       default=None,
                       help="Single testing CSV file")

train_arg.add_argument("--rolling_mean", type=int,
                       default=10,
                       help="Number of days used in calculating the rolling mean")
train_arg.add_argument("--input_length", type=int,
                       default=45,
                       help="Number of rolling mean input days to make a prediction on")
train_arg.add_argument("--output_length", type=int,
                       default=10,
                       help="Number of days predicted into the future")

train_arg.add_argument("--batch_size", type=int,
                       default=250,
                       help="Batch size for model input")
train_arg.add_argument("--epochs", type=int,
                       default=1000,
                       help="Number of epochs for training")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs/",
                       help="Location for saving logs")
train_arg.add_argument("--save_dir", type=str,
                       default="./save/",
                       help="Location for saving trained weights")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")
train_arg.add_argument("--val_intv", type=int,
                       default=2,
                       help="Validation interval")

model_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate")
model_arg.add_argument("--l2_reg", type=float,
                       default=0,
                       help="L2 Regularization strength")
model_arg.add_argument("--dropout", type=float,
                       default=0.2,
                       help="LSTM Dropout")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
