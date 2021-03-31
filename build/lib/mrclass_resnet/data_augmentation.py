from imgaug import augmenters as iaa


def get_aug_pipeline():
    aug_pipeline = iaa.Sequential([
        #iaa.Sometimes(0.5, iaa.GaussianBlur((0, 3.0))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
        # apply one of the augmentations: Dropout or CoarseDropout
        iaa.OneOf([
            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ]),
        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 3),[
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
            iaa.Fliplr(1.0), # horizontally flip
            iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))), # crop and pad 50% of the images
            iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
        ])
    ],
    random_order=True # apply the augmentations in random order
    )
    return aug_pipeline

# Apply augmentation pipeline to sample image
#images_aug = np.array([aug_pipeline.augment_image(image) for _ in range(16)])