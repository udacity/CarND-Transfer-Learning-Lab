# echo "Running Inception Cifar10"
# python run_bottleneck.py --network inception --batch_size 32 --dataset cifar10
# echo "Running Inception Traffic"
# python run_bottleneck.py --network inception --batch_size 32 --dataset traffic
echo "Running ResNet Cifar10"
python run_bottleneck.py --network resnet --batch_size 32 --dataset cifar10
echo "Running ResNet Traffic"
python run_bottleneck.py --network resnet --batch_size 32 --dataset traffic
echo "Running VGG Cifar10"
python run_bottleneck.py --network vgg --batch_size 16 --dataset cifar10
echo "Running VGG Traffic"
python run_bottleneck.py --network vgg --batch_size 16 --dataset traffic
