import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

from rt2qmd.qmd import QMDDump

class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(),
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, interval=50, **kwargs )

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '',
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        self.slider.set_val(i)


### using this class is as easy as using FuncAnimation:

PS_XZ = np.dtype([
    ('proton', bool),
    ('px'    , np.float32),
    ('pz'    , np.float32),
    ('x'     , np.float32),
    ('z'     , np.float32)
])

data = QMDDump('QMD_dump.bin')

idx_event = 3

# snapshot N:N+100 (N event, xz projection)
dim = data[0].model['current_field_size']

ps = np.empty((100, dim), dtype=PS_XZ)

for i in range(100):
    snapshot = data[i + idx_event*100]
    for j in range(dim):
        p = snapshot[j]
        ps[i][j]['proton'] = p.isProton()
        ps[i][j]['px']     = p.momentum()[0]
        ps[i][j]['pz']     = p.momentum()[2]
        ps[i][j]['x']      = p.position()[0]
        ps[i][j]['z']      = p.position()[2]

ps0_p = ps[0][ps[0]['proton'] == True]
ps0_n = ps[0][ps[0]['proton'] == False]

fig = plt.figure(figsize=(8,8))
ax  = fig.add_subplot(111)
pp, = ax.plot(ps0_p['z'], ps0_p['x'], "yo", ms=10, markerfacecolor='none')
pn, = ax.plot(ps0_n['z'], ps0_n['x'], "go", ms=10, markerfacecolor='none')
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.legend(['Proton', 'Neutron'])
plt.xlabel('z position [fm]')
plt.ylabel('x position [fm]')


def update(i):
    psi_p = ps[i][ps[i]['proton'] == True]
    psi_n = ps[i][ps[i]['proton'] == False]
    pp.set_data(psi_p['z'], psi_p['x'])
    pn.set_data(psi_n['z'], psi_n['x'])

ani = Player(fig, update, maxi=100 - 1)
ani.save('event.gif', writer='imagemagick', fps=15, dpi=100)

plt.show()
