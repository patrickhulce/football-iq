from engine import train_one_epoch, evaluate
import torchvision
import utils
from torchvision.transforms import v2 as T
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import torch

from torchvision.io import read_image
from torchvision import tv_tensors
from process_mask import process_mask


class FootballDataset(torch.utils.data.Dataset):
    def __init__(self, frame_directory, transforms):
        self.frame_directory = frame_directory
        self.transforms = transforms
        images = list(sorted(os.listdir(frame_directory)))
        self.imgs = [img for img in images if not "-mask" in img]
        self.masks = [img for img in images if img.endswith("-mask.png")]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.frame_directory, self.imgs[idx])
        mask_path = os.path.join(self.frame_directory, self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        target = process_mask(mask, idx)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has three classes only - background, player, football
    num_classes = 3
    # use our dataset and defined transformations
    dataset = FootballDataset('data/frames', get_transform(train=True))
    dataset_test = FootballDataset('data/frames', get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')

    print("That's it!")


if __name__ == '__main__':
    main()
