import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from tkinter import filedialog
from sigpy.util import prod, move


class Image(object):
    '''Plot numpy ndarray as image.
    Options can be selected through keypress

    Key options:
    ------------
    <x/y>        : select current dimension as x and y dimension
    <z>          : toggle current dimension as z dimension
    <t>          : swap between x and y axis
    <left/right> : increment/decrement current dimension
    <up/down>    : flip axis when current dimension is x or y
                   otherwise increment/decrement slice at current dimension
    <h>          : toggle hide all labels, titles and axes
    <m>          : magnitude mode
    <p>          : phase mode
    <r>          : real mode
    <i>          : imaginary mode
    <l>          : log mode
    '''

    def __init__(self, im, x=-1, y=-2, z=None, c=None, hide=False, mode='m', title='',
                 interpolation='lanczos', fps=10):
        if im.ndim < 2:
            raise TypeError('Image dimension must at least be two, got {im_ndim}'.format(
                im_ndim=im.ndim))

        self.im = im
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.shape = self.im.shape
        self.ndim = self.im.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.y = y % self.ndim
        self.z = z % self.ndim if z is not None else None
        self.c = c % self.ndim if c is not None else None
        self.d = max(self.ndim - 3, 0)
        self.hide = hide
        self.title = title
        self.interpolation = interpolation
        self.mode = mode
        self.axim = None
        self.entering_slice = False
        self.vmin = None
        self.vmax = None
        self.fps = fps

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == 'up':

            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'down':

            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'left':

            self.d = (self.d - 1) % self.ndim
            
            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'right':

            self.d = (self.d + 1) % self.ndim
            
            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'x' and self.d not in [self.x, self.z, self.c]:
            if self.d == self.y:
                self.x, self.y = self.y, self.x
            else:
                self.x = self.d
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'y' and self.d not in [self.y, self.z, self.c]:
            if self.d == self.x:
                self.x, self.y = self.y, self.x
            else:
                self.y = self.d
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'z' and self.d not in [self.x, self.y, self.c]:
            if self.d == self.z:
                self.z = None
            else:
                self.z = self.d
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif (event.key == 'c' and
              self.d not in [self.x, self.y, self.z] and
              self.shape[self.d] == 3):

            if self.d == self.c:
                self.c = None
            else:
                self.c = self.d
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 't':
            self.x, self.y = self.y, self.x
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'h':
            
            self.hide = not self.hide
                
            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'f':
            self.fig.canvas.manager.full_screen_toggle()
        
        elif (event.key == 'm' or event.key == 'p' or
              event.key == 'r' or event.key == 'i' or event.key == 'l'):

            self.vmin = None
            self.vmax = None
            self.mode = event.key
                
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 's':
            file_path = filedialog.asksaveasfilename(filetypes = (("png files","*.png"),
                                                                  ("pdf files","*.pdf"),
                                                                  ("eps files","*.eps"),
                                                                  ("svg files","*.svg"),
                                                                  ("jpeg files","*.jpg"),
                                                                  ("all files","*.*")))

            if not file_path:
                return
            
        elif event.key == 'v':
            file_path = filedialog.asksaveasfilename(filetypes = (("mp4 files","*.mp4"),
                                                                  ("all files","*.*")))
            
            if not file_path:
                return
            
            try:
                FFMpegWriter = ani.writers['ffmpeg']
            except:
                raise ValueError('Does not have FFMPEG installed.')

            writer = FFMpegWriter(fps=self.fps)

            with writer.saving(self.fig, file_path, 100):
                for i in range(self.shape[self.d]):
                    self.slices[self.d] = i

                    self.update_axes()
                    self.update_image()
                    self.fig.canvas.draw()
                    writer.grab_frame()

        elif (event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'backspace'] and
              self.d not in [self.x, self.y, self.z, self.c]):

            if self.entering_slice:
                if event.key == 'backspace':
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * 10 + int(event.key)
            else:
                self.entering_slice = True
                self.entered_slice = int(event.key)
                
            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'enter' and self.entering_slice:

            self.entering_slice = False
            if self.entered_slice < self.shape[self.d]:
                self.slices[self.d] = self.entered_slice    

                self.update_image()

            self.update_axes()
            self.fig.canvas.draw()
                
        else:
            return

    def update_image(self):

        idx = []
        for i in range(self.ndim):
            if i in [self.x, self.y, self.z, self.c]:
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])

        imv = move(self.im[idx])
        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims = [self.z] + imv_dims

        if self.c is not None:
            imv_dims = imv_dims + [self.c]

        imv = np.transpose(imv, np.argsort(imv_dims))
        imv = array_to_image(imv, color=self.c is not None)
        
        if self.mode == 'm':
            imv = np.abs(imv)
        elif self.mode == 'p':
            imv = np.angle(imv)
        elif self.mode == 'r':
            imv = np.real(imv)
        elif self.mode == 'i':
            imv = np.imag(imv)
        elif self.mode == 'l':
            eps = 1e-31
            imv = np.log(np.abs(imv) + eps)

        if self.vmin is None:
            self.vmin = imv.min()
            
        if self.vmax is None:
            self.vmax = imv.max()

        if self.axim is None:
            self.axim = self.ax.imshow(imv,
                                       vmin=self.vmin, vmax=self.vmax,
                                       cmap='gray', origin='lower',
                                       interpolation=self.interpolation, aspect=1.0,
                                       extent=[0, imv.shape[1], 0, imv.shape[0]])
            
        else:
            self.axim.set_data(imv)
            self.axim.set_extent([0, imv.shape[1], 0, imv.shape[0]])
            self.axim.set_clim(self.vmin, self.vmax)

    def update_axes(self):

        if not self.hide:
            caption = '['
            for i in range(self.ndim):

                if i == self.d:
                    caption += '['
                else:
                    caption += ' '

                if (self.flips[i] == -1 and (i == self.x or
                                             i == self.y or
                                             i == self.z)):
                    caption += '-'

                if i == self.x:
                    caption += 'x'
                elif i == self.y:
                    caption += 'y'
                elif i == self.z:
                    caption += 'z'
                elif i == self.c:
                    caption += 'c'
                elif i == self.d and self.entering_slice:
                    caption += str(self.entered_slice) + '_'
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += ']'
                else:
                    caption += ' '
            caption += ']'

            self.ax.set_title(caption)
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
        else:
            self.ax.set_title('')
            self.fig.suptitle('')
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)


def mosaic_shape(batch):

    mshape = [int(batch**0.5), batch // int(batch**0.5)]

    while (prod(mshape) < batch):
        mshape[1] += 1

    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1

    return tuple(mshape)


def array_to_image(arr, color=False):
    '''
    Flattens all dimensions except the last two
    '''

    if color:
        eps = 1e-31
        arr = arr / (np.abs(arr).max() + eps)
        
    if arr.ndim == 2:
        return arr
    elif color and arr.ndim == 3:
        return arr

    if color:
        ndim = 3
    else:
        ndim = 2
        
    shape = arr.shape
    batch = prod(shape[:-ndim])
    mshape = mosaic_shape(batch)

    if prod(mshape) == batch:
        img = arr.reshape((batch, ) + shape[-ndim:])
    else:
        img = np.zeros((prod(mshape), ) + shape[-ndim:], dtype=arr.dtype)
        img[:batch, ...] = arr.reshape((batch, ) + shape[-ndim:])

    img = img.reshape(mshape + shape[-ndim:])
    if color:
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = img.reshape((shape[-3] * mshape[-2], shape[-2] * mshape[-1], shape[-1]))
    else:
        img = np.transpose(img, (0, 2, 1, 3))
        img = img.reshape((shape[-2] * mshape[-2], shape[-1] * mshape[-1]))

    return img
