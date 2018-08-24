import os
import uuid
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sigpy.util import prod, move


class Image(object):
    """Plot array as image.

    Key press options:
        <x/y>: select current dimension as x and y dimension.
        <t>: swap between x and y axis.
        <z>: toggle current dimension as z dimension.
        <c>: toggle current dimension as color channel. 
            Only works if current dimension is of length 3.
        <left/right>: increment/decrement current dimension
        <up/down>: flip axis when current dimension is x or y.
            Otherwise increment/decrement slice at current dimension.
        <h>: toggle hide all labels, titles and axes.
        <m>: magnitude mode. Renormalizes when pressed each time.
        <p>: phase mode. Renormalizes when pressed each time.
        <r>: real mode. Renormalizes when pressed each time.
        <i>: imaginary mode. Renormalizes when pressed each time.
        <l>: log mode. Renormalizes when pressed each time.
        <s>: save as png.
        <g>: save as gif by traversing current dimension.
        <v>: save as mp4 by traversing current dimension.
        <0-9>: enter slice number.
        <enter>: Set current dimension as slice number.

    """
    def __init__(self, im, x=-1, y=-2, z=None, c=None, hide=False, mode='m', title='',
                 interpolation='nearest', save_basename='Figure', fps=10):
        if im.ndim < 2:
            raise TypeError('Image dimension must at least be two, got {im_ndim}'.format(
                im_ndim=im.ndim))

        self.axim = None
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
        self.entering_slice = False
        self.vmin = None
        self.vmax = None
        self.save_basename = save_basename
        self.fps = fps

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == 'up':
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (
                    self.slices[self.d] + 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'down':
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (
                    self.slices[self.d] - 1) % self.shape[self.d]
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
            filename = self.save_basename + \
                       datetime.datetime.now().strftime(' %Y-%m-%d at %h.%M.%S %p.png')
            self.fig.savefig(filename, transparent=True, format='png',
                             bbox_inches='tight', pad_inches=0)
            
        elif event.key == 'g':
            filename = self.save_basename + \
                       datetime.datetime.now().strftime(' %Y-%m-%d at %h.%M.%S %p.gif')
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig('{} {:05d}.png'.format(temp_basename, i),
                                 format='png', bbox_inches=bbox, pad_inches=0)
                
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi),
                                                 int(bbox.height * self.fig.dpi)),
                            '-r', str(self.fps),
                            '-i', '{} %05d.png'.format(temp_basename),
                            '-vf', 'palettegen', '{} palette.png'.format(temp_basename)])
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi),
                                                 int(bbox.height * self.fig.dpi)),
                            '-r', str(self.fps),
                            '-i', '{} %05d.png'.format(temp_basename),
                            '-i', '{} palette.png'.format(temp_basename),
                            '-lavfi', 'paletteuse', filename])
            
            os.remove('{} palette.png'.format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove('{} {:05d}.png'.format(temp_basename, i))
            
        elif event.key == 'v':
            filename = self.save_basename + \
                       datetime.datetime.now().strftime(' %Y-%m-%d at %h.%M.%S %p.mp4')
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig('{} {:05d}.png'.format(temp_basename, i),
                                 format='png', bbox_inches=bbox, pad_inches=0)
                
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi),
                                                 int(bbox.height * self.fig.dpi)),
                            '-r', str(self.fps),
                            '-i', '{} %05d.png'.format(temp_basename),
                            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', filename])
            
            for i in range(self.shape[self.d]):
                os.remove('{} {:05d}.png'.format(temp_basename, i))

        elif (event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'backspace'] and
              self.d not in [self.x, self.y, self.z, self.c]):

            if self.entering_slice:
                if event.key == 'backspace':
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * \
                        10 + int(event.key)
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
        # Extract slice.
        idx = []
        for i in range(self.ndim):
            if i in [self.x, self.y, self.z, self.c]:
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])

        imv = move(self.im[idx])

        # Transpose to have [z, y, x, c].
        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims = [self.z] + imv_dims

        if self.c is not None:
            imv_dims = imv_dims + [self.c]

        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
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
            imv = np.log(np.abs(imv), out=np.ones_like(imv) * np.infty, where=imv != 0)

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
    """
    Flattens all dimensions except the last two

    """
    if color:
        arr = np.divide(arr, np.abs(arr).max(),
                        out=np.zeros_like(arr), where=arr != 0)

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
        img = img.reshape(
            (shape[-3] * mshape[-2], shape[-2] * mshape[-1], shape[-1]))
    else:
        img = np.transpose(img, (0, 2, 1, 3))
        img = img.reshape((shape[-2] * mshape[-2], shape[-1] * mshape[-1]))

    return img
