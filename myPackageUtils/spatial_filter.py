import numpy as np
import numpy.fft as fft
import numpy as np

# filter spatial features
def spatial_filter(d, char_len):
    """
    Function that computes the spatial frequency low-pass filter thus enabling to have a smoother vector field as output if the input is stochastic gaussian noise
    """
    # compute the bidimensional space fft
    if char_len >= 2:
        d_fft = fft.fftshift(fft.fft2(d))
        (x_dim, y_dim) = (d.shape[0], d.shape[1])
        (x_center, y_center) = (x_dim//2, y_dim//2)
        
        # mask of rough filter section
        # Filter only low frequencies
        R = d.shape[0]/(char_len)
        mask = np.zeros_like(d, dtype=np.double)
        for i in range(x_dim):
            for j in range(y_dim):
                # compute rexpect to center
                x_c = i-x_center
                y_c = j-y_center
                if ( (x_c**2 + y_c**2) < R**2 ):
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0
        
        # Filtered signal
        filtered = d_fft * mask
        out = fft.ifft2(fft.ifftshift(filtered))
    else:
        out = d
    
    return out
    