
import numpy as np
import re
import matplotlib.pyplot as plt
import sympy as sp

# Funcion para actualizar textos
def update(inp,txt):
    newtxt = str(inp.get())
    txt.config(text=newtxt)

# Funcion para validar que la entrada sea numerica y solo dos decimales
def validate_numeric_input(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^-?\d*\.?\d{0,2}$', value_if_allowed):
            return True
        else:
            return False
    else:
        return True

# Funcion para validar que la entrada sea numerica con más decimales
def validate_numeric_input_more_decimals(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^-?\d*\.?\d*$', value_if_allowed):
            return True
        else:
            return False
    else:
        return True
    
# Funcion para construir la viga y las cargas en una matrix 2xn donde 
# beam[0][n] es el punto en la viga
# beam[1][n] es la carga en cada punto
# beam[2][n] es la carga distribuida en cada punto
# beam[3][n] es el momento en cada punto
# beam[4][n] es la fuerza cortante en cada punto
# beam[5][n] es el momento flector en cada punto
# beam[6][n] es la inclinacion en cada punto
# beam[7][n] es la deflexion en cada punto

def build_beam(len):
    global scale, step
    step = 0.01
    scale = 1/step
    dom = np.arange(0, len+step, step)
    loads = np.zeros_like(dom)
    distloads = np.zeros_like(dom)
    moment = np.zeros_like(dom)
    shear = np.zeros_like(dom)
    flex = np.zeros_like(dom)
    slope = np.zeros_like(dom)
    deflection = np.zeros_like(dom)
    beam = np.array([dom, loads, distloads, moment, shear, flex, slope, deflection])
    return beam

# Funcion para desactivar los botones de los apoyos

def update_disable(inp,txt,type,apoyo1,apoyo2,apo1button,apo2button):
    update(inp,txt)
    global beamtype, supp1
    if type.get() == 'Cantilever':
        apoyo1.config(state='disabled')
        apoyo2.config(state='disabled')
        apo1button.config(state='disabled')
        apo2button.config(state='disabled')
        beamtype = 1
        supp1 = 0
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
    global beam, scale
    x = int(float(pos.get())*scale)
    beam[1][x] -= float(mag.get())
    return beam

# Funcion que ubica los momentos

def point_moment(pos,mag):
    global beam, scale
    x = int(float(pos.get())*scale)
    beam[3][x] += float(mag.get())
    return beam

# Funcion que ubica las cargas distribuidas

def distributed_load(start, end, mag):
    global beam, scale
    xstart = int(float(start.get())*scale)
    xend = int(float(end.get())*scale)
    x = sp.symbols('x')
    equation = sp.sympify(mag.get())

    for i in range(0,xend-xstart+1):
        pos = i+xstart
        beam[2][pos] -= equation.subs(x,i*step)
    return beam
        

# Funcion que calcula las reacciones en los apoyos

def calculate_reactions():
    global beam, supp1, supp2, beamtype, R1, R2, MR, scale
    R1 = 0
    R2 = 0
    M1 = 0
    M2 = 0
    MR = 0
    if beamtype == 1:
        for i in range(len(beam[0])):
            MR -= beam[0][i]*(beam[1][i]-beam[2][i]) + beam[3][i]
            R1 -= beam[1][i]+beam[2][i]
        beam[1][0] = R1
        beam[3][0] = MR
        print(R1, MR)
    elif beamtype == 2:
        for i in range(len(beam[0])):
            M1 += (beam[0][i]-supp1/scale)*(beam[1][i]+beam[2][i]) + beam[3][i]
            M2 += (beam[0][i]-supp2/scale)*(beam[1][i]+beam[2][i]) + beam[3][i]
            R1 = -M2/(supp1/scale-supp2/scale)
            R2 = -M1/(supp2/scale-supp1/scale)
        beam[1][supp1] = R1
        beam[1][supp2] = R2
   
    
    
    return R1, R2,  MR

# Funcion que grafica la viga

def plot_beam(beamgraph,figbeam):
    global beam, supp1, supp2, scale, beamtype
    figbeam.clear()
    ax = figbeam.add_subplot(111)
    if beamtype == 1:
        ax.plot([0,0],[5,-1],linewidth=5, color='gray', zorder=101)
        
    elif beamtype == 2:
        ax.plot([supp1/scale,supp1/scale],[0,-1],linewidth=3, color='red')
        ax.plot([supp2/scale,supp2/scale],[0,-1],linewidth=3, color='red')
        
    ax.plot(beam[0], np.zeros_like(beam[0]), linewidth=5, zorder=100)
    ax.set_ylim(-1,5.5)
    ax.yaxis.set_visible(False) 

    maxpoint = np.min(beam[1])
    maxdist = np.min(beam[2])
    truemax = min(maxpoint,maxdist)
    sizing  = 5/abs(truemax)

    for i in range(len(beam[0])):
        if beam[1][i] < 0:
            ratio = beam[1][i]/truemax
            xstart = i/scale
            ystart = 5*(ratio)+0.3
            xcomponent = 0
            ycomponent = -4*ratio
            width = ratio*0.05*len(beam[0])/scale
            height = ratio
            ax.arrow(xstart,ystart,xcomponent,ycomponent,head_width=width,head_length=height,linewidth=3, fc='k',ec='k')
    ax.plot(beam[0], -beam[2]*sizing, color='blue')
    ax.fill_between(beam[0], -beam[2]*sizing, 0, color='blue', alpha=0.5, interpolate=True)
    plt.show()
    beamgraph.draw()

    # Funcion que grafica los cortantes 

def plot_shear(sheargraph,figshear):
    global beam, step
    figshear.clear()
    ax = figshear.add_subplot(111)
    beam[4][0] = beam[1][0] + beam[2][0]
    for i in range(len(beam[0])-1):
        beam[4][i+1] = beam[4][i] + step*beam[1][i+1] + step*beam[2][i+1]

    max_shear = beam[4][np.argmax(np.abs(beam[4]))]
    max_shear_pos = beam[0][np.argmax(np.abs(beam[4]))]

    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[4][0]],color='blue')
    ax.plot(beam[0], beam[4],color='blue')
    ax.title.set_text('Fuerzas cortantes')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Fuerza cortante (N)')
    ax.fill_between(beam[0], beam[4], 0, color='blue', alpha=0.5, interpolate=True) 
    ax.plot(max_shear_pos, max_shear, 'r*', markersize=10, label=f'Fuerza cortante máxima: {round(max_shear,2)} N')
    ax.legend()
    sheargraph.draw()

def plot_moment(momentgraph,figmoment):
    global beam, scale, step
    figmoment.clear()
    ax = figmoment.add_subplot(111)
    beam[5][0] = beam[1][0] - beam[3][0]
    for i in range(len(beam[0])-1):
        beam[5][i+1] = beam[5][i] + step*beam[4][i+1] - beam[3][i+1]

    max_moment = beam[5][np.argmax(np.abs(beam[5]))]
    max_moment_pos = beam[0][np.argmax(np.abs(beam[5]))]

    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[5][0]], color='orange')
    ax.plot(beam[0], beam[5], color='orange')
    ax.title.set_text('Momento flector')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Momento flector (Nm)')
    ax.fill_between(beam[0], beam[5], 0, color='orange', alpha=0.5) 
    ax.plot(max_moment_pos, max_moment, 'r*', markersize=10, label=f'Momento flexor máximo: {round(max_moment,2)} Nm')
    ax.legend()
    momentgraph.draw()

def add_youngs_modulus(inp):
    global E
    E = float(inp.get())

def add_inertia(inp):
    global I
    I = float(inp.get())

def plot_slope(slopegraph,figslope):
    global beam, E, I, scale, step
    figslope.clear()
    ax = figslope.add_subplot(111)
    beam[6][0] = beam[5][0]/(E*I)
    for i in range(len(beam[0])-1):
        beam[6][i+1] = beam[6][i] + step*beam[5][i+1]/(E*I)

    max_slope = beam[6][np.argmax(np.abs(beam[6]))]
    max_slope_pos = beam[0][np.argmax(np.abs(beam[6]))]
    
    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[6][0]], color='green')
    ax.plot(beam[0], beam[6], color='green')
    ax.title.set_text('Inclinación de la viga')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Inclinación (rad)')
    ax.fill_between(beam[0], beam[6], 0, color='green', alpha=0.5)
    ax.plot(max_slope_pos, max_slope, 'r*', markersize=10, label=f'Inclinación máxima: {max_slope} rad')
    ax.legend()
    slopegraph.draw()

def plot_deflection(deflectiongraph,figdeflection):
    global beam, E, I, scale, step
    figdeflection.clear()
    ax = figdeflection.add_subplot(111)
    beam[7][0] = beam[6][0]
    for i in range(len(beam[0])-1):
        beam[7][i+1] = beam[7][i] + step*beam[6][i+1]
    
    max_deflection = beam[7][np.argmax(np.abs(beam[7]))]
    max_deflection_pos = beam[0][np.argmax(np.abs(beam[7]))]

    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam[7][0]], color='purple')
    ax.plot(beam[0], beam[7], color='purple')
    ax.title.set_text('Deflexión de la viga')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Deflexión (m)')
    ax.fill_between(beam[0], beam[7], 0, color='purple', alpha=0.5)
    ax.plot(max_deflection_pos, max_deflection, 'r*', markersize=10, label=f'Deflexión máxima: {round(max_deflection,2)} m')
    ax.legend()
    deflectiongraph.draw()

# Funcion que llama todas las funciones para calcular la viga

def calculate_beam(beamgraph,figbeam,sheargraph,figshear,momentgraph,
figmoment,slopegraph,figslope,deflectiongraph,figdeflection):
    global beam
    beam[4] = np.zeros_like(beam[0])
    beam[5] = np.zeros_like(beam[0])
    beam[6] = np.zeros_like(beam[0])
    beam[7] = np.zeros_like(beam[0])
    calculate_reactions()
    plot_beam(beamgraph,figbeam)
    plot_shear(sheargraph,figshear)
    plot_moment(momentgraph,figmoment)
    plot_slope(slopegraph,figslope)
    plot_deflection(deflectiongraph,figdeflection)

def reset_all(entries,figures):
    global beam
    beam = build_beam(0)
    for entry in entries:
        entry.delete(0,'end')
    for figure in figures:
        figure.clear()
