
class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--nepoch', type=int, default=100, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')


        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='1', help='GPUs')
        parser.add_argument('--arch', type=str, default='unet', help='archtechture')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
        parser.add_argument('--seed', type=int, default=1234, help='dd_in')
        parser.add_argument("--best_iou_threshold",type=float,default=0.5,help="best_iou_threshold")
        parser.add_argument("--SMOOTH", type=float, default=1e-6, help="best_iou_threshold")
        parser.add_argument("--train_iterations", type=int, default=300, help="train_iterations")#300
        parser.add_argument("--print_iter", type=int, default=100, help="print_iter")#100
        parser.add_argument("--val_iterations", type=int, default=40, help="val_iterations")#40
        parser.add_argument("--description", type=str, default="seed1234", help="generate the result with extra information")  # 40
        # args for data
        parser.add_argument('--save_dir', type=str, default='/data/faultdetect/thebe', help='save dir')
        parser.add_argument('--data_dir', type=str, default='/data/faultseg', help='dir of data')

        parser.add_argument('--data_rate',type=float, default=0.5, help='to detect data efficient, range:(0,1)')

        parser.add_argument('--resume', action='store_true', default=False)

        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs for warmup')

        # ddp
        parser.add_argument("--local_rank", type=int, default=-1,
                            help='DDP parameter, do not modify')
        parser.add_argument("--distribute", action='store_true', help='whether using multi gpu train')
        parser.add_argument("--distribute_mode", type=str, default='DDP', help="using which mode to ")
        return parser