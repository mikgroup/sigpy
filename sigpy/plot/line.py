import os
import uuid
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sigpy.util import prod


class Line(object):
    """Plot array as lines.

    Key options:
        <x>: select current dimension as x
        <left/right>: increment/decrement current dimension
        <up/down>: flip axis when current dimension is x or y
            otherwise increment/decrement slice at current dimension
        <h>: toggle hide all labels, titles and axes
        <m>: magnitude mode
        <p>: phase mode
        <r>: real mode
        <i>: imaginary mode
        <l>: log mode
        <s>: save as png.
        <g>: save as gif by traversing current dimension.
        <v>: save as mp4 by traversing current dimension.
    """

    def __init__(self, arr, x=-1, hide=False, mode='m', title='',
                 save_basename='Figure', fps=10):
        self.arr = arr
        self.axarr = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.shape = self.arr.shape
        self.ndim = self.arr.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.d = max(self.ndim - 3, 0)
        self.hide = hide
        self.title = title
        self.mode = mode
        self.save_basename = save_basename
        self.fps = fps

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_line()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == 'up':
            if self.d != self.x:
                self.slices[self.d] = (
                    self.slices[self.d] + 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == 'down':

            if self.d != self.x:
                self.slices[self.d] = (
                    self.slices[self.d] - 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == 'left':

            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'right':

            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'x' and self.d != self.x:

            self.x = self.d

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == 'h':
            self.hide = not self.hide

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == 'f':
            self.fig.canvas.manager.full_screen_toggle()

        elif (event.key == 'm' or event.key == 'p' or
              event.key == 'r' or event.key == 'i' or event.key == 'l'):

            self.mode = event.key

            self.update_axes()
            self.update_line()
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
                self.update_line()
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
                self.update_line()
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
        else:
            return

        return

    def update_line(self):

        order = ([i for i in range(self.ndim)
                  if i != self.x] + [self.x])
        idx = ([self.slices[i] for i in order[:-1]] +
               [slice(None, None, self.flips[self.x])])

        arrv = self.arr.transpose(order)[idx]

        if self.mode == 'm':
            arrv = np.abs(arrv)
        elif self.mode == 'a':
            arrv = np.angle(arrv)
        elif self.mode == 'r':
            arrv = np.real(arrv)
        elif self.mode == 'i':
            arrv = np.imag(arrv)
        elif self.mode == 'l':
            eps = 1e-31
            arrv = np.log(np.abs(arrv) + eps)

        if self.axarr is None:
            self.axarr = self.ax.plot(arrv)[0]

        else:
            self.axarr.set_xdata(np.arange(len(arrv)))
            self.axarr.set_ydata(arrv)
            self.ax.relim()
            self.ax.autoscale_view()

    def update_axes(self):

        if not self.hide:
            caption = 'Slice: ['
            for i in range(self.ndim):

                if i == self.d:
                    caption += '['
                else:
                    caption += ' '

                if self.flips[i] == -1 and i == self.x:
                    caption += '-'

                if i == self.x:
                    caption += 'x'
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += ']'
                else:
                    caption += ' '
            caption += ']'

            self.ax.set_title(caption)
            self.ax.axis('on')
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
