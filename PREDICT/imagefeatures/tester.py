import SimpleITK as sitk
# import shape_features as sf
# import vessel_features as vf
# import log_features as lf
# import image_helper as ih
# from texture_features import get_NGTDM_features as gf
from phase_features import get_phase_features as gf
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology
import scipy.ndimage as nd


def ExtractNLargestBlobsn(binaryImage, numberToExtract=1):
    """Extract N largest blobs from binary image.

    Arguments:
        binaryImage: boolean numpy array one or several shape_masks.
        numberToExtract: number of blobs to extract (integer).

    Returns:
        binaryImage: boolean numpy are containing only the N
                     extracted blobs.
    """

    # Get all the blob properties.
    labeledImage = label(binaryImage, connectivity=3)
    blobMeasurements = regionprops(labeledImage)

    if len(blobMeasurements) == 1:
        # Single blob, return input
        binaryImage = binaryImage
    else:
        # Get all the areas
        allAreas = list()
        allCoords = list()
        for blob in blobMeasurements:
            allAreas.append(blob.area)
            allCoords.append(blob.coords)

        allAreas = np.asarray(allAreas)
        if numberToExtract > 0:
            # For positive numbers, sort in order of largest to smallest.
            # Sort them.
            indices = np.argsort(-allAreas)
        elif numberToExtract < 0:
            # For negative numbers, sort in order of smallest to largest.
            # Sort them.
            indices = np.argsort(allAreas)
        else:
            raise ValueError("Number of blobs to extract should not be zero!")

        binaryImage = np.zeros(binaryImage.shape)
        # NOTE: There must be a more efficient way to do this
        for nblob in range(0, numberToExtract):
            nblob = abs(nblob)
            coords = allCoords[indices[nblob]]
            for coord in coords:
                binaryImage[coord[0], coord[1], coord[2]] = 1

    return binaryImage


# mask = sitk.ReadImage('/media/martijn/DATA/IC/compact_data/1000_seg_0_all.nii.gz')
# image = sitk.ReadImage('/media/martijn/DATA/BLT/BLTRadiomics-004/BLTRadiomics-004_2/10/image.nii.gz')
# mask = sitk.ReadImage('/media/martijn/DATA/BLT/BLTRadiomics-004/BLTRadiomics-004_2/10/seg_v1_Razvan_20170331_1938.nii.gz')

image = sitk.ReadImage('/media/martijn/DATA/BLT/BLTRadiomics-007/BLTRadiomics-007/8/image.nii.gz')
mask = sitk.ReadImage('/media/martijn/DATA/BLT/BLTRadiomics-007/BLTRadiomics-007/8/seg_v1_Razvan_20170321_1940.nii.gz')

# shape_mask = ih.get_masked_slices_mask(mask)
shape_mask = sitk.GetArrayFromImage(mask)
shape_mask = nd.binary_fill_holes(shape_mask)
shape_mask = morphology.remove_small_objects(shape_mask, min_size=2, connectivity=2, in_place=False)
shape_mask = shape_mask.astype(bool)
shape_mask = ExtractNLargestBlobsn(shape_mask, 1)
shape_mask = shape_mask.astype(np.uint8)
shape_mask = sitk.GetImageFromArray(shape_mask)
# shape_mask[shape_mask > 1] = 1
# shape_mask = sitk.GetImageFromArray(shape_mask)


# print("Computing shape features.")
# f, l = sf.get_shape_features_old(shape_mask)
# print f
# print l

# f1, l1 = sf.get_shape_features_2D_old(shape_mask)
# print f1
# print l1
# f2, l2 = sf.get_shape_features(shape_mask, None, '2D')
# print f2
# print l2
# f = [f11 - f22 for f11, f22 in zip(f1,f2)]
# print f
# f, l = sf.get_shape_features(shape_mask, None, '3D')
# print f
# print l

# print("Computing vessel features.")
# image = sitk.GetArrayFromImage(image)
# shape_mask = sitk.GetArrayFromImage(shape_mask)
# image = np.transpose(image, [2, 1, 0])
# shape_mask = np.transpose(shape_mask, [2, 1, 0])
# f1, l1 = vf.get_vessel_features(image, shape_mask)
# print f1
# print l1

# print("Computing log features.")
# image = sitk.GetArrayFromImage(image)
# shape_mask = sitk.GetArrayFromImage(shape_mask)
# image = np.transpose(image, [2, 1, 0])
# shape_mask = np.transpose(shape_mask, [2, 1, 0])
# f1, l1 = lf.get_log_features(image, shape_mask)
# print f1
# print l1

# print("Computing NGTDM features.")
# image = sitk.GetArrayFromImage(image)
# shape_mask = sitk.GetArrayFromImage(shape_mask)
# image = np.transpose(image, [2, 1, 0])
# shape_mask = np.transpose(shape_mask, [2, 1, 0])
# f1, l1 = gf(image, shape_mask)
# print f1
# print l1

print("Computing Phase features.")
image = sitk.GetArrayFromImage(image)
shape_mask = sitk.GetArrayFromImage(shape_mask)
image = np.transpose(image, [2, 1, 0])
shape_mask = np.transpose(shape_mask, [2, 1, 0])
f1, l1 = gf(image, shape_mask)
print f1
print l1
