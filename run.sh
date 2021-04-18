# mnist
## baseline
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.0 --beta 0.0 --n_epoch 30 --num_classes 10
## symmetric noise
### 0.2
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.4
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.6
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.8
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 10
## asymmetric noise
### 0.1
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.2
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.3
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.4
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 10


# cifar10
## baseline
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.0 --beta 0.0 --n_epoch 30 --num_classes 10
## symmetric noise
### 0.2
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.4
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.6
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.8
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 10
## asymmetric noise
### 0.1
python main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.2
python main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.3
python main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 10
### 0.4
python main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 10


# cifar100
## baseline
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.0 --beta 0.0 --n_epoch 30 --num_classes 100
## symmetric noise
### 0.2
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.4
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.6
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.8
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 100
## asymmetric noise
### 0.1
python main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.2 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.2
python main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.4 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.3
python main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.6 --beta 0.8 --n_epoch 30 --num_classes 100
### 0.4
python main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.8 --beta 0.8 --n_epoch 30 --num_classes 100