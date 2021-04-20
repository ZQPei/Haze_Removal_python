import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter
from numba import jit
import matplotlib.pyplot as plt

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        pass

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double)/255.
        # self.gray = np.array(img.convert('L'))
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)
        

    @staticmethod
    @jit(nopython=True)
    def _get_dark_channel(rows,cols,tmp,dark, radius=7):

        for i in range(rows):
            for j in range(cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,cols-1)
                dark[i,j] = tmp[rmin:rmax+1,cmin:cmax+1].min()
        return dark

    def get_dark_channel(self, radius=7):
        tmp = self.src.min(axis=2)
        # get_dark_channel[blockspergrid, threadsperblock]()
        # self.dark=self._get_dark_channel[self.blockspergrid, self.threadsperblock](self.rows,self.cols,tmp,self.dark,radius)
        self.dark=self._get_dark_channel(self.rows,self.cols,tmp,self.dark,radius)

    def get_air_light(self):
        print("Starting to compute air light prior...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows*self.cols*0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark>=threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:,:].mean(axis=0)
        # print(self.Alight)
        print("time:",time.time()-start)

    @staticmethod
    @jit(nopython=True)
    def _get_transmission(rows,cols,Alight,src,tran,radius=7, omega=0.95):

        for i in range(rows):
            for j in range(cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,cols-1)
                pixel = (src[rmin:rmax+1,cmin:cmax+1]/Alight).min()
                tran[i,j] = 1. - omega * pixel
        return tran

    def get_transmission(self, radius=7, omega=0.95):
        self.tran=self._get_transmission(self.rows,self.cols,self.Alight,self.src,self.tran,radius,omega)

    def guided_filter(self, r=60, eps=0.001):
        print("Starting to compute guided filter trainsmission...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, r, eps)
        print("time:",time.time()-start)

    def recover(self, t0=0.1):
        print("Starting recovering...")
        start = time.time()
        self.gtran[self.gtran<t0] = t0
        t = self.gtran.reshape(*self.gtran.shape,1).repeat(3,axis=2)
        # import ipdb; ipdb.set_trace()
        self.dst = (self.src.astype(np.double) - self.Alight)/t + self.Alight
        self.dst *= 255
        self.dst[self.dst>255] = 255
        self.dst[self.dst<0] = 0
        self.dst = self.dst.astype(np.uint8)
        print("time:",time.time()-start)

    def show(self):
        import cv2
        cv2.imwrite("img/src.jpg", (self.src*255).astype(np.uint8)[:,:,(2,1,0)])
        cv2.imwrite("img/dark.jpg", (self.dark*255).astype(np.uint8))
        cv2.imwrite("img/tran.jpg", (self.tran*255).astype(np.uint8))
        cv2.imwrite("img/gtran.jpg", (self.gtran*255).astype(np.uint8))
        cv2.imwrite("img/dst.jpg", self.dst[:,:,(2,1,0)])
        
        io.imsave("test.jpg", self.dst)



if __name__ == '__main__':
    import sys
    hr = HazeRemoval()
    hr.open_image(sys.argv[1])
    hr.get_dark_channel()
    hr.get_air_light()
    hr.get_transmission()
    hr.guided_filter()
    hr.recover()
    hr.show()

    