from parse_config import args
from data.data_load import DataLoad_MNIST_CIFAR
from models.CNN import CNN_MNIST, CNN_CIFAR10, ResNet, BasicBlock
from models.noise_layer import NoiseLayer
from utils import Baseline_train, test, train_predict, Hybrid_train

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


def main():
    # write file: file name
    file_name_prefix = 'results/' + str(args.dataset) + '_' + str(args.noise_type)
    baseline_model_path = 'baseline_checkpoints/' + str(args.dataset) + '.pkl'

    if args.beta == 0.0 or args.noise_rate == 0.0:  # beta=0, noise_rate=0; only baseline
        # clean dataset processing
        clean_train_loader, clean_test_loader = DataLoad_MNIST_CIFAR(args.dataset, noise_rate=0.0)
        # model preparation
        if args.dataset == "mnist":
            Baseline_model = CNN_MNIST(1, args.num_classes)
        elif args.dataset == 'cifar10':
            Baseline_model = CNN_CIFAR10(3, args.num_classes)
        elif args.dataset == 'cifar100':
            Baseline_model = ResNet(BasicBlock, [2, 2, 2, 2])
        # Baseline model, loss function
        Baseline_model = Baseline_model.cuda(args.gpu)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = optim.Adam(Baseline_model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        # Baseline train and test
        for epoch in range(args.n_epoch):
            print('Baseline Epoch:', epoch)
            baseline_train_acc_list, baseline_train_loss_list = Baseline_train(clean_train_loader, Baseline_model,
                                                                               optimizer, criterion)
            baseline_test_acc_list, baseline_test_loss_list = test(clean_test_loader, Baseline_model, criterion)
        print("Finished Baseline training.")
        # write into csv file
        # print(len(baseline_train_acc_list), len(baseline_train_loss_list), len(baseline_test_acc_list), len(baseline_test_loss_list))
        df = pd.DataFrame.from_dict({'baseline_train_acc': baseline_train_acc_list,
                                     'baseline_train_loss': baseline_train_loss_list,
                                     'baseline_test_acc': baseline_test_acc_list,
                                     'baseline_test_loss': baseline_test_loss_list}, orient='index')
        df = df.transpose()
        df.to_csv(file_name_prefix + str(args.noise_rate) +'.csv')
        # save baseline model
        torch.save(Baseline_model.state_dict(), baseline_model_path)
    else:  # noise model training
        # noise_test_loader is not injected with noise, just for name difference
        noise_train_loader, noise_test_loader = DataLoad_MNIST_CIFAR(args.dataset, noise_rate=args.noise_rate)
        # model preparation, load model
        if args.dataset == "mnist":
            Baseline_model = CNN_MNIST(1, args.num_classes)
        elif args.dataset == 'cifar10':
            Baseline_model = CNN_CIFAR10(3, args.num_classes)
        elif args.dataset == 'cifar100':
            Baseline_model = ResNet(BasicBlock, [2, 2, 2, 2])
        Baseline_model.load_state_dict(torch.load(baseline_model_path))
        # Baseline model, loss function
        Baseline_model = Baseline_model.cuda(args.gpu)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = optim.Adam(Baseline_model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        # build confusion matrix
        Baseline_output, y_train_noise = train_predict(Baseline_model, noise_train_loader)
        Baseline_confusion = np.zeros((args.num_classes, args.num_classes))
        for n, p in zip(y_train_noise, Baseline_output):
            n = n.cpu().numpy()
            p = p.cpu().numpy()
            Baseline_confusion[p, n] += 1.
        # noisy channel
        channel_weights = Baseline_confusion.copy()
        channel_weights /= channel_weights.sum(axis=1, keepdims=True)
        channel_weights = np.log(channel_weights + 1e-8)
        channel_weights = torch.from_numpy(channel_weights)  # numpy.ndarray -> tensor
        channel_weights = channel_weights.float()
        noisemodel = NoiseLayer(theta=channel_weights.cuda(args.gpu), k=args.num_classes)
        noise_optimizer = optim.Adam(noisemodel.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        print("noisy channel finished.")
        # noisy model train and test
        for epoch in range(args.n_epoch):
            print('Revision Epoch:', epoch)
            noise_train_acc_list, noise_train_loss_list = Hybrid_train(noise_train_loader, Baseline_model, noisemodel,
                                                                       optimizer, noise_optimizer, criterion)
            print("After hybrid, test acc: ")
            noise_test_acc_list, noise_test_loss_list = test(noise_test_loader, Baseline_model, criterion)
        print("Finished hybrid training.")
        # write into csv file
        df = pd.DataFrame.from_dict({str(args.noise_rate) + '_noise_train_acc': noise_train_acc_list,
                           str(args.noise_rate) + '_noise_train_loss': noise_train_loss_list,
                           str(args.noise_rate) + '_noise_test_acc': noise_test_acc_list,
                           str(args.noise_rate) + '_noise_test_loss': noise_test_loss_list}, orient='index')
        df = df.transpose()
        df.to_csv(file_name_prefix + str(args.noise_rate) + '.csv')


if __name__ == '__main__':
    main()
