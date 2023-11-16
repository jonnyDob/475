import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()

        self.image_paths = []
        self.labels = []
        self.transform = transform

        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # self.transform = transforms.Compose([
        #     transforms.Resize(size=(size,size), interpolation=Image.BICUBIC),
        #     # transforms.RandomCrop(crop_size),
        #     transforms.ToTensor()
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # this must be the lines text file
        infoFilePath = dir + "/labels.txt"
        with open(infoFilePath, 'r') as file:
            lines = file.readlines()

        for line in lines:

            parts = line.split()
            imgName = parts[0]
            imgLabel = int(parts[1])

            imgPath = f"{dir}/{imgName}"

            self.image_paths.append(imgPath)
            self.labels.append(imgLabel)

    # TODO fix whatever this underline is
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image_sample = self.transform(image)

        # image = Image.open(self.image_files[index]).convert('RGB')
        # image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)

        return image_sample, self.labels[index]
