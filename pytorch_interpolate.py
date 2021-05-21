import torch

def interp_bilinear(img, idx_x, idx_y):
    """
    Bilinear interpolation
    input:
        img - [C,H,W] tensor from which values are interpolated
        idx_x - [h,w] tensor of (x)-indices corresponding to locations in img to be interpolated,
        idx_y - [h,w] tensor of (y)-indices corresponding to locations in img to be interpolated,
            the values idx_x[:,:] must be between 0 and W-1,
            the values idx_y[:,:] must be between 0 and H-1.
    output:
        img_interp - [C,h,w]
    """
    # Notation inspired by wikipedia ( https://en.wikipedia.org/w/index.php?title=Bilinear_interpolation&oldid=1018246009 ): 
    idx_x_1 = idx_x.floor().long()
    idx_x_2 = idx_x.ceil().long()
    idx_y_1 = idx_y.floor().long()
    idx_y_2 = idx_y.ceil().long()
    img_interp = (idx_x_2 - idx_x)*(idx_y_2 - idx_y)*img[:, idx_y_1, idx_x_1]
    img_interp += (idx_x - idx_x_1)*(idx_y_2 - idx_y)*img[:, idx_y_1, idx_x_2]
    img_interp += (idx_x_2 - idx_x)*(idx_y - idx_y_1)*img[:, idx_y_2, idx_x_1]
    img_interp += (idx_x - idx_x_1)*(idx_y - idx_y_1)*img[:, idx_y_2, idx_x_2]
    return img_interp

def interp_nearest_nb(img, idx_x, idx_y):
    """
    Nearest neighbour interpolation
    input:
        img - [C,H,W] tensor from which values are interpolated
        idx_x - [h,w] tensor of (x)-indices corresponding to locations in img to be interpolated,
        idx_y - [h,w] tensor of (y)-indices corresponding to locations in img to be interpolated,
            the values idx_x[:,:] must be between 0 and W-1,
            the values idx_y[:,:] must be between 0 and H-1.
    output:
        img_interp - [C,h,w]
    """
    idx_x = idx_x.round().long()
    idx_y = idx_y.round().long()
    img_interp = img[:, idx_y, idx_x]
    return img_interp
