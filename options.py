import argparse


class Options():
    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--batch_size_train', type=int, default=14)
        parser.add_argument('--batch_size_val', type=int, default=14)
        parser.add_argument('--patch_size_train', type=int, default=512)
        parser.add_argument('--patch_size_val', type=int, default=512)
        parser.add_argument('--train_input', type=str, default='')
        parser.add_argument('--train_target', type=str, default='')
        parser.add_argument('--val_input', type=str, default='')
        parser.add_argument('--val_target', type=str, default='')
        parser.add_argument('--test_input', type=str, default='')
        parser.add_argument('--test_target', type=str, default='')
        parser.add_argument('--result_path', type=str, default='./results/')
        parser.add_argument('--save_dir', type=str, default='./')
        parser.add_argument('--pretrained', type=str, default='')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--val_freq', type=int, default=3)
        parser.add_argument('--save_freq', type=int, default=20)
        parser.add_argument('--no_cuda', action='store_true')
        args = parser.parse_args()

        self.Epoch = args.epochs
        self.Learning_Rate = args.lr
        self.Batch_Size_Train = args.batch_size_train
        self.Batch_Size_Val = args.batch_size_val
        self.Patch_Size_Train = args.patch_size_train
        self.Patch_Size_Val = args.patch_size_val
        self.Input_Path_Train = args.train_input
        self.Target_Path_Train = args.train_target
        self.Input_Path_Val = args.val_input
        self.Target_Path_Val = args.val_target
        self.Input_Path_Test = args.test_input
        self.Target_Path_Test = args.test_target
        self.Result_Path_Test = args.result_path
        self.MODEL_SAVE_PATH = args.save_dir
        self.MODEL_PRE_PATH = args.pretrained
        self.Num_Works = args.num_workers
        self.Val_Freq = args.val_freq
        self.Save_Freq = args.save_freq
        self.CUDA_USE = not args.no_cuda
