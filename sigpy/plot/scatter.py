import os
import uuid
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sigpy.util import prod, move


class Scatter(object):
    """Plot array as image.

    Key press options:
        <z>: toggle current dimension as z dimension
        <left/right>: increment/decrement current dimension
        <up/down>: flip axis when current dimension is x or y
            otherwise increment/decrement slice at current dimension
        <h>: toggle hide all labels, titles and axes
        <m>: magnitude mode
        <p>: phase mode
        <r>: real mode
        <i>: imaginary mode
        <l>: log mode
    """

    def __init__(self, coord, data=None, z=None, hide=False, mode='m', title='',
                 save_basename='Figure', fps=10):

        self.coord = coord
        assert coord.shape[-1] == 2
        if data is None:
            self.data = np.ones(coord.shape[:-1])
        else:
            self.data = data

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('k')

        for c, d in zip(coord.shape[:-1], self.data.shape[-coord.ndim + 1:]):
            assert c == d

        self.ndim = self.data.ndim - self.coord.ndim + 1
        self.shape = self.data.shape[:self.ndim]

        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.z = z % self.ndim if z is not None else None
        self.d = 0
        self.hide = hide
        self.title = title
        self.mode = mode
        self.axsc = None
        self.entering_slice = False
        self.vmin = None
        self.vmax = None
        self.save_basename = save_basename
        self.fps = fsp

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_data()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == 'up':
            if self.d != self.z:
                self.slices[self.d] = (
                    self.slices[self.d] + 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_data()
            self.fig.canvas.draw()

        elif event.key == 'down':
            if self.d != self.z:
                self.slices[self.d] = (
                    self.slices[self.d] - 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_data()
            self.fig.canvas.draw()

        elif event.key == 'left':
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'right':
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        # elif event.key == 'z':
        #     if self.d == self.z:
        #         self.z = None
        #     else:
        #         self.z = self.d

        #     self.update_axes()
        #     self.update_data()
        #     self.fig.canvas.draw()

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
            self.update_data()
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
                self.update_data()
                self.fig.canvas.draw()
                self.fig.savefig('{} {:05d}.png'.format(temp_basename, i),
                                 format='png', bbox_inches=bbox, pad_inches=0)
                
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi), int(bbox.height * self.fig.dpi)),
                            '-r', str(self.fps),
                            '-i', '{} %05d.png'.format(temp_basename),
                            '-vf', 'palettegen', '{} palette.png'.format(temp_basename)])
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi), int(bbox.height * self.fig.dpi)),
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
                self.update_data()
                self.fig.canvas.draw()
                self.fig.savefig('{} {:05d}.png'.format(temp_basename, i),
                                 format='png', bbox_inches=bbox, pad_inches=0)
                
            subprocess.run(['ffmpeg', '-f', 'image2',
                            '-s', '{}x{}'.format(int(bbox.width * self.fig.dpi), int(bbox.height * self.fig.dpi)),
                            '-r', str(self.fps),
                            '-i', '{} %05d.png'.format(temp_basename),
                            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', filename])
            
            for i in range(self.shape[self.d]):
                os.remove('{} {:05d}.png'.format(temp_basename, i))

        elif (event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'backspace'] and
              self.d != self.z):

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

                self.update_data()

            self.update_axes()
            self.fig.canvas.draw()

        else:
            return

    def update_data(self):

        idx = []
        for i in range(self.ndim):
            if i == self.z:
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])

        datav = move(self.data[idx])
        # if self.z is not None:
        #     datav_dims = [self.z] + datav_dims
        coordv = move(self.coord)

        if self.mode == 'm':
            datav = np.abs(datav)
        elif self.mode == 'p':
            datav = np.angle(datav)
        elif self.mode == 'r':
            datav = np.real(datav)
        elif self.mode == 'i':
            datav = np.imag(datav)
        elif self.mode == 'l':
            eps = 1e-31
            datav = np.log(np.abs(datav) + eps)

        datav = datav.ravel()

        if self.axsc is None:
            self.axsc = self.ax.scatter(
                coordv[..., 0].ravel(), coordv[..., 1].ravel(), c=datav,
                s=1, linewidths=0, cmap='gray',
                vmin=self.vmin, vmax=self.vmax,
            )

        else:
            self.axsc.set_offsets(coordv.T.reshape([-1, 2]))
            self.axsc.set_color(datav)

    def update_axes(self):

        if not self.hide:
            caption = '['
            for i in range(self.ndim):

                if i == self.d:
                    caption += '['
                else:
                    caption += ' '

                if (self.flips[i] == -1 and i == self.z):
                    caption += '-'

                if i == self.z:
                    caption += 'z'
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
