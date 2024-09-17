
import numpy as np
import re
import matplotlib.pyplot as plt

# Funcion para actualizar textos
def update(inp,txt):
    newtxt = str(inp.get())
    txt.config(text=newtxt)

# Funcion para validar que la entrada sea numerica y solo dos decimales
def validate_numeric_input(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^\d*\.?\d{0,2}$', value_if_allowed):
            return True
        else:
            return False
    else:
        return True
    
# Funcion para construir la viga y las cargas en una matrix 2xn donde 
# beam[0][n] es el punto en la viga
# beam[1][n] es la carga en cada punto
# beam[2][n] es el momento en cada punto
# beam[3][n] es la fuerza cortante en cada punto
# beam[4][n] es el momento flector en cada punto

def build_beam(len):
    global scale
    step = 0.001
    scale = 1/step
    dom = np.arange(0, len+0.01, step)
    loads = np.zeros_like(dom)
    moment = np.zeros_like(dom)
    shear = np.zeros_like(dom)
    flex = np.zeros_like(dom)
    beam = np.array([dom, loads,moment,shear,flex])
    return beam

# Funcion para desactivar los botones de los apoyos

def update_disable(inp,txt,type,apoyo1,apoyo2,apo1button,apo2button):
    update(inp,txt)
    global beamtype 
    if type.get() == 'Cantilever':
        apoyo1.config(state='disabled')
        apoyo2.config(state='disabled')
        apo1button.config(state='disabled')
        apo2button.config(state='disabled')
        beamtype = 1
    else:
        apoyo1.config(state='normal')
        apoyo2.config(state='normal')
        apo1button.config(state='normal')
        apo2button.config(state='normal')
        beamtype = 2

# Funcion que llama a la funcion build_beam y update

def update_build(inp,txt):
    update(inp,txt)
    global beam
    beam = build_beam(float(inp.get()))

# Funcion que llama a la funcion update y ubica los soportes

def update_supports(inp,txt,support):
    update(inp,txt)
    global supp1, supp2, scale
    if support == 1:
        supp1 = int(float(inp.get())*scale)
    elif support == 2:
        supp2 = int(float(inp.get())*scale)

# Funcion que ubica las cargas

def point_load(pos,mag):
    global beam
    x = int(float(pos.get())*scale)
    beam[1][x] -= float(mag.get())
    return beam

# Funcion que ubica los momentos

def point_moment(pos,mag):
    global beam, scale
    x = int(float(pos.get())*scale)
    beam[2][x] += float(mag.get())
    return beam

# Funcion que calcula las reacciones en los apoyos

def calculate_reactions():
    global beam, supp1, supp2, beamtype, R1, R2, scale
    R1 = 0
    R2 = 0
    M1 = 0
    M2 = 0
    if beamtype == 1:
        for i in range(len(beam[0])):
            R1 -= beam[1][i]
    elif beamtype == 2:
        for i in range(len(beam[0])):
            M1 += (beam[0][i]-supp1/scale)*beam[1][i]
            M2 += (beam[0][i]-supp2/scale)*beam[1][i]
            R1 = -M2/(supp1/scale-supp2/scale)
            R2 = -M1/(supp2/scale-supp1/scale)
    beam[1][supp1] = R1
    beam[1][supp2] = R2
    
    print(R1,R2)
    return R1, R2

# Funcion que grafica la viga

def plot_beam(beamgraph,figbeam):
    global beam, supp1, supp2, scale
    figbeam.clear()
    ax = figbeam.add_subplot(111)
    ax.plot(beam[0], np.zeros_like(beam[0]), linewidth=5, zorder=100)
    ax.plot([supp1/scale,supp1/scale],[0,-1],linewidth=3, color='red')
    ax.plot([supp2/scale,supp2/scale],[0,-1],linewidth=3, color='red')
    ax.set_ylim(-1,5.5)
    ax.yaxis.set_visible(False) 

    max = np.min(beam[1])

    for i in range(len(beam[0])):
        if beam[1][i] < 0:
            ratio = beam[1][i]/max
            xstart = i/scale
            ystart = 5*(ratio)+0.3
            xcomponent = 0
            ycomponent = -4*ratio
            width = ratio*0.05*len(beam[0])/scale
            height = ratio
            ax.arrow(xstart,ystart,xcomponent,ycomponent,head_width=width,head_length=height,linewidth=3, fc='k',ec='k')
    plt.show()
    beamgraph.draw()

    # Funcion que grafica los cortantes 

def plot_shear(sheargraph,figshear):
    global beam
    figshear.clear()
    ax = figshear.add_subplot(111)
    beam[3][0] = beam[1][0]
    for i in range(len(beam[0])-1):
        beam[3][i+1] = beam[3][i] + beam[1][i+1]

    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[3][0]],color='blue')
    ax.plot(beam[0], beam[3],color='blue')
    ax.fill_between(beam[0], beam[3], 0, color='blue', alpha=0.5, interpolate=True) 
    sheargraph.draw()

def plot_moment(momentgraph,figmoment):
    global beam
    figmoment.clear()
    ax = figmoment.add_subplot(111)
    beam[4][0] = beam[1][0] 
    for i in range(len(beam[0])-1):
        beam[4][i+1] = beam[4][i] + beam[3][i+1]

    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[4][0]], color='orange')
    ax.plot(beam[0], beam[4], color='orange')
    ax.fill_between(beam[0], beam[4], 0, color='orange', alpha=0.5) 
    momentgraph.draw()

    
        


# Funcion que llama todas las funciones para calcular la viga

def calculate_beam(beamgraph,figbeam,sheargraph,figshear,momentgraph,figmoment):
    calculate_reactions()
    plot_beam(beamgraph,figbeam)
    plot_shear(sheargraph,figshear)
    plot_moment(momentgraph,figmoment)
