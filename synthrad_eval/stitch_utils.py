# Vendored from https://github.com/SynthRAD2025/evaluation @ 99e9ef4 (stitch_utils.py, GPL-3.0).
# Changes are marked "synthrad_eval:"; everything else is the organisers' code, byte for byte.

import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # synthrad_eval: headless


def resample_reference(image, ref_image, default_value=0, log=None)->sitk.Image:
    """
    Resamples the given image to the grid of a reference image.

    Parameters:
    - image: SimpleITK.Image
        The input struct to be resampled.
    - ref_image: SimpleITK.Image
        reference image for resampling.

    Returns:
    - resampled_image: SimpleITK.Image
        The resampled struct.

    """
    # Create a resample filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetDefaultPixelValue(default_value)
    # Apply the resampling
    resampled_image = resampler.Execute(image)
    
    if log != None:
        log.info(f'Struct resampled to reference image!')
    return resampled_image


def stitch_image_noresample(image_inside: sitk.Image, image_outside: sitk.Image, mask: sitk.Image) -> sitk.Image:
    image_inside_np = sitk.GetArrayFromImage(image_inside) 
    mask_np = sitk.GetArrayFromImage(mask)
    image_outside_np = sitk.GetArrayFromImage(image_outside)

    image_stitched_np = image_outside_np * (mask_np == 0) + image_inside_np * (mask_np > 0)
    image_stitched = sitk.GetImageFromArray(image_stitched_np)
    image_stitched.CopyInformation(image_outside)
    return image_stitched

def stitch_image(image_inside: sitk.Image, image_outside: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    Stitches the `image_inside` and `image_outside` based on the `mask`.
    Also resamples the inside image and mask to be on same grid as outside image.

    Args:
        image_inside (sitk.Image): The image to be stitched inside the mask.
        image_outside (sitk.Image): The image to be stitched outside the mask.
        mask (sitk.Image): The mask used to determine the stitching region.

    Returns:
        sitk.Image: The stitched image.

    """
    print(image_inside.GetSpacing(), image_inside.GetOrigin(), image_inside.GetDirection(), image_inside.GetSize())
    print(image_outside.GetSpacing(), image_outside.GetOrigin(), image_outside.GetDirection(), image_outside.GetSize())
    print(mask.GetSpacing(), mask.GetOrigin(), mask.GetDirection(), mask.GetSize())

    image_inside = resample_reference(image_inside,image_outside,default_value=0)
    mask = resample_reference(mask,image_outside,default_value=0)
    mask = sitk.BinaryErode(mask, (1, 1, 1))


    image_inside_np = sitk.GetArrayFromImage(image_inside) 
    mask_np = sitk.GetArrayFromImage(mask)
    image_outside_np = sitk.GetArrayFromImage(image_outside)
    plt.imshow(image_inside_np[60,...])
    plt.figure()
    plt.imshow(mask_np[60,...])
    plt.figure()

    plt.imshow(image_outside_np[60,...])


    # stitch images
    image_stitched_np = image_outside_np * (mask_np == 0) + image_inside_np * (mask_np > 0)
    image_stitched = sitk.GetImageFromArray(image_stitched_np)
    image_stitched.CopyInformation(image_outside)
    plt.figure()
    plt.imshow(image_stitched_np[60,...])
    # plt.show()

    return image_stitched

