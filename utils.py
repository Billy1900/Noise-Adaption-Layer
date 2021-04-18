import torch
from parse_config import args


def Baseline_train(train_loader, model, optimizer, criterion):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    baseline_loss = []
    baseline_acc = []
    for batch_x, batch_y in train_loader:
        # send the training data to the GPU
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # set all gradient to zero
        optimizer.zero_grad()
        # forward propagation
        out = model(batch_x)
        # calculate loss and acc
        loss = criterion(out, batch_y)
        acc = accuracy(out, batch_y)
        train_acc_meter.update(acc, out.size(0))
        train_loss_meter.update(loss, out.size(0))
        # log acc and loss
        baseline_acc.append(train_acc_meter.val)
        baseline_loss.append(train_loss_meter.val.cpu().detach().numpy())
        # back propogation
        loss.backward()
        # update the parameters (weights and biases)
        optimizer.step()
    print('# Train || Loss : %.4f , Acc : %.4f%%' % (train_loss_meter.val, train_acc_meter.val * 100.0))
    return baseline_acc, baseline_loss


def Hybrid_train(train_loader, model, noisemodel, optimizer, noise_optimizer, criterion):
    model.train()
    noisemodel.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    noise_acc_list = []
    noise_loss_list = []
    for batch_x, batch_y in train_loader:
        # send the training data to the GPU
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # set all gradient to zero
        optimizer.zero_grad()
        noise_optimizer.zero_grad()
        # forward propagation
        out = model(batch_x)
        predictions = noisemodel(out)
        # calculate loss and acc
        Baseline_loss = criterion(out, batch_y)
        noisemodel_loss = criterion(predictions, batch_y)
        loss = (1 - args.beta) * Baseline_loss + args.beta * noisemodel_loss
        acc = accuracy(predictions, batch_y)
        train_acc_meter.update(acc, out.size(0))
        train_loss_meter.update(loss, out.size(0))
        # log acc and loss
        noise_acc_list.append(train_acc_meter.val)
        noise_loss_list.append(train_loss_meter.val.cpu().detach().numpy())
        # back propogation
        loss.backward()
        # update the parameters (weights and biases)
        optimizer.step()
        noise_optimizer.step()
    print('# Hybrid Train || Loss : %.4f , Acc : %.4f%% ' % (train_loss_meter.val, train_acc_meter.val * 100.0))
    return noise_acc_list, noise_loss_list


def test(test_loader, model, criterion):
    model.eval()
    test_loss_meter = AverageMeter()
    test_acc_meter = AverageMeter()
    test_loss_list = []
    test_acc_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            # forward propagation
            out = model(batch_x)
            # calculate loss and acc
            loss = criterion(out, batch_y)
            acc = accuracy(out, batch_y)
            test_acc_meter.update(acc, out.size(0))
            test_loss_meter.update(loss, out.size(0))
            # log acc and loss
            test_acc_list.append(test_acc_meter.val)
            test_loss_list.append(test_loss_meter.val.cpu().detach().numpy())
    print('# Evaluation || Loss : %.4f, Acc : %.4f%%' % (test_loss_meter.val, test_acc_meter.val * 100.0))
    return test_acc_list, test_loss_list


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    correct = 0
    _, pred = output.max(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / target.size(0)
    return acc


def train_predict(model, train_loader):
    model.eval()
    prediction_list = []
    y_train_noise = []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            out = model(batch_x)
            _, predicted = torch.max(out, 1)
            prediction_list.append(predicted)
            y_train_noise.append(batch_y)
    return prediction_list, y_train_noise
