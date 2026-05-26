#this module inherits the torch Dataset class to create pairs of images and labels on the fly during training
#it picks the images from the image directory and pairs them up. sometimes images of the same person labelled as 1 and
#images of different persons labelled as 0

#making imports
import random
from pathlib import Path 
from typing import Dict, List, Tuple, Optional, Set

#torch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

#src
from src.logger import get_logger
from src.utils import set_seed

logger = get_logger(__name__)

class SiameseDataset(Dataset):
    """
    Dataset that returns pairs of face images and a similarity label (1 = same person, 0 = different).
    Pairs are generated on-the-fly with a 50% chance of being positive.
    """
    def __init__(self, root_dir:str, person_to_image_dict: Dict[str, List[str]], transform=None, allowed_persons: Optional[Set[str]]=None):
        """
        Args:
            root_dir: Directory containing person subfolders.
            person_to_images: Mapping from person name to list of image file names.
            transform: torchvision transform to apply.
            allowed_persons: If given, restrict the dataset to these identities.
        """
        self.root_dir = root_dir
        self.person_to_image_dict = person_to_image_dict
        self.transform = transform
        self.allowed_persons = allowed_persons

        #build a flat list of (person, image_path)
        self.samples: List[Tuple[str,str]] = [] #[(person, img1,img2,...),(person, img1, img2,...)]
        for person, img_pths in self.person_to_image_dict.items():
            if self.allowed_persons and person not in self.allowed_persons:
                continue
            for image_name in img_pths:
                self.samples.append((person,image_name))

        #build a lookup for same person images
        self.person_to_images: Dict[str, List[str]] = {}
        for person, img_pths in self.person_to_image_dict.items():
            if self.allowed_persons and person not in self.allowed_persons:
                continue
            self.person_to_images[person] = img_pths
        
        #people with at least 2 images-this is because we want positve pairs or pairs with label=1
        self.multi_images_per_person = [p for p , imgs in self.person_to_images.items() if len(imgs) >= 2]

        logger.info(
            "Dataset created with %d samples from %d identities (multi-image: %d)",
            len(self.samples),
            len(self.person_to_images),
            len(self.multi_images_per_person)
        )

    def __len__(self)->int:
        return len(self.samples)
    def __getitem__(self, index:int)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        person1, img1_name = self.samples[index]

        #decide pair type
        if person1 in self.multi_images_per_person and random.random() < 0.5:
            #positive pair: same person, different image
            img2_name = random.choice(self.person_to_images[person1])
            attempts=0
            while img2_name == img1_name and len(self.person_to_images[person1]) > 1:
                img2_name = random.choice(self.person_to_images[person1])
                attempts+=1
                if attempts > 10:
                    break
                label=1.0
                person2  = person1 #this helps as set person2 to avoid value error
        else:
            #negative pair: different person
            person2 = random.choice([p for p in self.person_to_images if p != person1])
            img2_name = random.choice(self.person_to_images[person2])
            label = 0.0

        #load images
        img1 = Image.open(self.root_dir / person1/ img1_name).convert("RGB")
        img2 = Image.open(self.root_dir / person2 / img2_name).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor([label], dtype=torch.float32)
    
def create_dataset(config:Dict)->Tuple[SiameseDataset, SiameseDataset, SiameseDataset]:
    """
    Build the training and validation SiameseDataset instances using the config.
    Returns (train_dataset, val_dataset).
    """
    #expand cache dir and find image root
    cache_dir = Path(config["data"]["cache_dir"]).expanduser()
    root_dir = cache_dir / "lfw"/"lfw-deepfunneled"/"lfw-deepfunneled"
    if not root_dir.exists():
        raise FileNotFoundError(f"Data root not found: {root_dir}. Run data ingestion first or check image directory tree")
    allowed_extensions = set(config["data"]["allowed_extensions"]) #see config.yaml for allowed extension of images or to update them
    image_size = config["data"]["image_size"]

    #build person -> image list mapping
    person_to_images : Dict[str, List[str]] = {}
    for person_dir in root_dir.iterdir() if isinstance(root_dir, Path) else Path(root_dir):
        if not person_dir.is_dir():
            continue
        images = [
            f.name for f in person_dir.iterdir() if f.suffix.lower() in allowed_extensions
        ]
        if images:
            person_to_images[person_dir.name] = images

    logger.info("found %d idenities in %s", len(person_to_images), root_dir)

    #split identities
    all_persons = list(person_to_images.keys())
    random.seed(config["split"]["seed"])
    random.shuffle(all_persons)

    #ratios to consider
    train_ratio = config["split"].get("train_ratio",0.72)
    val_ratio = config["split"].get("val_ratio",0.18)
    test_ratio = config["split"].get("test_ratio",0.10)

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"split ratios sum to {total_ratio:.3f}, but must sum to 1.0 for all ratios (test, val, train)")
    n_total = len(all_persons)
    train_end = int(train_ratio * n_total)
    val_end = train_end + int(val_ratio * n_total)

    train_persons = set(all_persons[:train_end])
    val_persons = set(all_persons[train_end:val_end])
    test_persons = set(all_persons[val_end:]) if test_ratio > 0 else set()

    #debug
    logger.info(f"train_persons: {len(train_persons)}, val_persons: {len(val_persons)} test_persons: {len(test_persons)}")

    #transforms (same for all splits)
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    #create train, val, test sets
    train_dataset = SiameseDataset(root_dir=root_dir, person_to_image_dict=person_to_images, transform=transform,allowed_persons=train_persons)
    val_dataset = SiameseDataset(root_dir=root_dir, person_to_image_dict=person_to_images, transform=transform,allowed_persons=val_persons)
    test_dataset = SiameseDataset(root_dir=root_dir, person_to_image_dict=person_to_images, transform=transform,allowed_persons=test_persons) if test_persons else None

    logger.info(
        "splits: train=%d, val=%s, test=%d identities",
        len(train_dataset), len(val_dataset), len(test_dataset) 
    )
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    from src.utils import load_config
    from src.data_extractor.dataset import SiameseDataset

    config = load_config(config_path="./config.yaml")

    create_dataset(config=config)