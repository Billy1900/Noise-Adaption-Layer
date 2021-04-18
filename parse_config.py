import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.001)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--beta', type=float, default=0.8, help='BETA how much weight to give to the 2nd softmax loss and '
                                                            'BETA for the standard/baseline 1st softmax;'
                                                            'if beta=0, only baseline will run')
parser.add_argument('--noise_type', type=str, default='pairflip', help='pairflip or symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--gpu', default=0)

args = parser.parse_args()