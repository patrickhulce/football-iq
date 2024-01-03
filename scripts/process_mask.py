import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def process_mask(mask, image_id):
    # instances are encoded as different colors
    # background is black
    # football is white
    # each player is a different color
    mask = mask.permute(1, 2, 0)
    mask_pixel_list = mask.reshape(-1, 3)
    unique_colors = torch.unique(mask_pixel_list, dim=0)

    background_color = torch.tensor([0, 0, 0])
    player_colors_bitmask = ~(unique_colors > 250).all(
        dim=1) & ~(unique_colors == background_color).all(dim=1)
    player_colors = unique_colors[player_colors_bitmask]

    print("Mask shape: ", mask.shape)

    football_mask = (mask > 250).all(dim=2)
    player_masks = torch.stack(
        [(mask == player_color).all(dim=2) for player_color in player_colors])

    # split the color-encoded mask into a set
    # of binary masks
    has_football_mask = False
    masks = player_masks

    if (football_mask == True).any():
        print("Football mask found!!")
        has_football_mask = True
        masks = torch.cat([football_mask.unsqueeze(0), masks], dim=0)

    num_objs = len(masks)

    print("Masks shape: ", masks.shape)

    # get bounding box coordinates for each mask
    boxes = masks_to_boxes(masks)

    # the football is white, and every other color is a player
    labels = torch.ones((num_objs,), dtype=torch.int64)
    if has_football_mask:
        labels[0] = 2

    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {}
    target["boxes"] = tv_tensors.BoundingBoxes(
        boxes, format="XYXY", canvas_size=F.get_size(mask))
    target["masks"] = tv_tensors.Mask(masks)
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    return target
