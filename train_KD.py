import argparse
import os
import os.path as osp
import torch



def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/workspace/DBs/CC/ProcessedData/', help='data path')
    parser.add_argument('--dataset', default='shha', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--teacher_ckpt', default='exp/teacher_model_shha/best_model_1.pth', type=str,
                        help='the path of load teacher model')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--wfeature', type=float, default=0.1, help='weight on feature loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')
    parser.add_argument('--trainer', type=str, default='teacher', help='teacher / student')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
        args.data_dir = osp.join(args.data_dir, 'QNRF')
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'shha':
        from datasets.SHHA.loading_data import loading_data
        args.crop_size = 256
        args.data_dir = osp.join(args.data_dir, 'Shanghai_A')
    elif args.dataset.lower() == 'shhb':
        args.crop_size = 512
        args.data_dir = osp.join(args.data_dir, 'Shanghai_B')
    else:
        raise NotImplementedError
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = loading_data(args)

    if args.trainer == 'teacher':
        from teacher import Trainer
    elif args.trainer == 'meta':
        from meta_student import Trainer
    else:
        from student import Trainer
    trainer = Trainer(args, device, dataloaders)
    trainer.setup()
    trainer.train()
