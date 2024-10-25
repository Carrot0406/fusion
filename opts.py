import argparse


def opt_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_rgb', type=str, default=r'/home/bailu/fusion/dataset/28GHz/RGB',
                        help='indicator to dataset')
    parser.add_argument('--dataset_cloud', type=str, default=r'/home/bailu/fusion/dataset/28GHz/lidar',
                        help='indicator to dataset')
    parser.add_argument('--dataset_depth', type=str,
                        default=r'/home/bailu/fusion/dataset/28GHz/depth',
                        help='indicator to dataset')
    parser.add_argument('--dataset_pl', type=str, default='', help='indicator to dataset')

    # path setting
    parser.add_argument('--result_path', type=str, default=r'/home/bailu/fusion/result', help='path to the folder to save results')
    parser.add_argument('--method', type=str, default='concat', help='')
    # parser.add_argument('--method', type=str, default='attention_only', help='')
    # experiment controls
    parser.add_argument('--modality', type=str, default='v+c', help='choose modality for experiment: v, s, v+s')
    parser.add_argument('--mode', type=str, default='train',
                        help='select from train, val, test. Used in dataset creation')
    parser.add_argument('--net_v', type=str, default='resnet50',
                        help='choose network backbone for image channel: vgg19bn, resnet18, resnet50, wrn, wiser')
    parser.add_argument('--net_s', type=str, default='pointnet')
    # turning parameters

    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_finetune', type=float, default=1e-4, help='fine-tune learning rate')
    parser.add_argument('--lrd_rate', type=float, default=0.08, help='decay rate of learning rate')
    parser.add_argument('--lrd_rate_finetune', type=float, default=0.1, help='decay rate of fine-tune learning rate')
    parser.add_argument('--lr_decay', type=int, default=4, help='decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')

    args = parser.parse_args()

    return args