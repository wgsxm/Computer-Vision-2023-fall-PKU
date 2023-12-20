import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter

PATH = './model.pt'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_loss_function():
    return F.cross_entropy

def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(degrees=30),  
        transforms.Resize((args.size,args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # create dataloader

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # create optimizer

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # create scheduler 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # ctreat summary writer

    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_dir = f'./log/{current_time}'
    writer = SummaryWriter(log_dir)

    model.to(device)
    model.train()
    criterion = create_loss_function()

    for epoch in range(args.num_epochs):
        # train
        running_loss = 0
        temp_loss = 0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            # optimize
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()
            temp_loss += loss.item()
            if i % 200 == 199:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {temp_loss / 200:.3f}')
                temp_loss = 0.0

        # scheduler adjusts learning rate
        scheduler.step()
        # test
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the {total} test images: {accuracy} %')

        model.train()
        average_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', average_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
    writer.flush()
    writer.close()
    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, PATH)

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    
    model.to(device)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # create testing dataset
    transform_test = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # create dataloader
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # test
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # forward
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy} %')
        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--run', type=str, default='train', help='Train or Test')
    parser.add_argument('--model', type=str, default='VGG', help='Type of the chosen model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--size', type=int, default=60, help='Size of a picture')
    args = parser.parse_args()

    if args.model == 'VGG':
        model = VGG(args.num_classes)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
        }, PATH)
    exit()
    if args.run == 'train':
        train(model, args)
    # train / test