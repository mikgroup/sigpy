import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from tkinter import filedialog
from sigpy.util import prod


class Line(object):
    '''Plot numpy ndarray as lines.
    Options can be selected through keypress

    Key options:
    ------------
    <x>        : select current dimension as x
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

    def __init__(self, arr, x=-1, hide=False, mode='m', title=''):
        self.arr = arr

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
        self.axarr = None

        slices = [0] * (self.ndim - 1) + [slice(None)]
        if mode == 'm':
            arrv = np.abs(self.arr[slices])
        elif mode == 'p':
            arrv = np.angle(self.arr[slices])
        elif mode == 'r':
            arrv = np.real(self.arr[slices])
        elif mode == 'i':
            arrv = np.imag(self.arr[slices])
        elif self.mode == 'l':
            eps = 1e-31
            arrv = np.log(np.abs(self.arr[slices]) + eps)

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_line()
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
            file_path = filedialog.asksaveasfilename(filetypes=(("png files", "*.png"),
                                                                ("pdf files",
                                                                 "*.pdf"),
                                                                ("eps files",
                                                                 "*.eps"),
                                                                ("svg files",
                                                                 "*.svg"),
                                                                ("jpeg files",
                                                                 "*.jpg"),
                                                                ("all files", "*.*")))

            if not file_path:
                return

        elif event.key == 'v':
            file_path = filedialog.asksaveasfilename(filetypes=(("mp4 files", "*.mp4"),
                                                                ("all files", "*.*")))

            if not file_path:
                return

            try:
                FFMpegWriter = ani.writers['ffmpeg']
            except:
                raise ValueError('Does not have FFMPEG installed.')

            writer = FFMpegWriter(fps=10)

            with writer.saving(self.fig, file_path, 100):
                for i in range(self.shape[self.d]):
                    self.slices[self.d] = i

                    self.update_axes()
                    self.update_image()
                    self.fig.canvas.draw()
                    writer.grab_frame()

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
