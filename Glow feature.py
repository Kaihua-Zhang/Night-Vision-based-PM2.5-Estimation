import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def DarkChannel(im, sz):
    """
    :param im: image
    :param sz: kernel size
    :return:   dark channel
    """
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    dark = cv2.erode(dc, kernel)

    return dark


def AtmLight(im, sz):
    """
    :param im: image
    :param sz: kernel size
    :return:   airlight const
    """
    b, g, r = cv2.split(im)  # 取出通道
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    r = cv2.dilate(r, kernel)
    g = cv2.dilate(g, kernel)
    b = cv2.dilate(b, kernel)
    light = cv2.merge([b, g, r])

    return light


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / (A[:, :, ind] + 0.0001)
    transmission = 1 - omega * DarkChannel(im3, sz)

    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255.
    r    = 60
    eps  = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def get_D_Te_Tg(img):
    src = img
    I = img.astype('float64') / 255.
    dark = DarkChannel(I, 15)       # 得到暗通道
    A = AtmLight(I, 15)
    te = TransmissionEstimate(I, A, 15)
    t_guided = TransmissionRefine(src, te)
    return dark, te, t_guided


def new_GLOW(R,B,t):
    R = AtmLight(R, 15)
    G = (R * (B / 255.) * (1 - t / 255.)).clip(0, 255)
    return G


def brightness(R):
    max = np.max(R, axis=2) / 255.
    min = np.min(R, axis=2) / 255.
    B = (np.power((max - min) + max, 3).clip(0, 1) * 255)
    return B

def plot(A, B, C, D):
    A = A[:,:,::-1].astype(np.uint8)
    B = B[:,:,::-1].astype(np.uint8)
    C = C[:,:,::-1].astype(np.uint8)
    D = D[:,:,::-1].astype(np.uint8)
    plt.figure("Features")  # 图像窗口名称
    #
    plt.subplot(221)
    plt.imshow(A)
    plt.axis('off')
    plt.title('Raw Image')
    #
    plt.subplot(222)
    plt.imshow(B, plt.cm.gray)
    plt.axis('off')
    plt.title('Transmission Map')
    #
    plt.subplot(223)
    plt.imshow(C, plt.cm.gray)
    plt.axis('off')
    plt.title('Light Source Estimation Map')
    #
    plt.subplot(224)
    plt.imshow(D)
    plt.axis('off')
    plt.title('Glow Map')

    plt.show()


def Glow(path):
    R = cv2.imread(path)
    B = brightness(R)
    D, te, T = get_D_Te_Tg(R)
    T = (T * 255).clip(0, 255)

    T = np.expand_dims(T, axis=2)
    B = np.expand_dims(B, axis=2)
    G = new_GLOW(R, B, T)

    plot(R, T, B, G)



if __name__ == "__main__":
    main_dir = os.path.join( os.getcwd(), "Fig" )
    for img in os.listdir(main_dir):
        path = os.path.join( main_dir, img )
        Glow(path)