
import numpy as np
import re
import matplotlib.pyplot as plt
import sympy as sp
import tkinter as tk
from tkinter import ttk
import math
import matplotlib.patches as patches

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

# Funcion para validar que la entrada sea numerica
def validate_numeric_input_more_decimals(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^-?\d*\.?\d*$', value_if_allowed):
            return True
        else:
            return False    
    else:
        return True
    
# Funcion para construir la viga y las cargas en una matrix 2xn donde n es la cantidad de puntos en la viga

def build_beam(len):
    global scale, step
    order = math.floor(math.log10(len))
    step = 10 ** (order-4)
    scale = 1/step
    dom = np.arange(0, len+step, step)  # beam[0][n] es el punto en la viga
    loads = np.zeros_like(dom)          # beam[1][n] es la carga puntual en cada punto
    distloads = np.zeros_like(dom)      # beam[2][n] es la carga distribuida en cada punto
    moment = np.zeros_like(dom)         # beam[3][n] es el momento en cada punto
    shear = np.zeros_like(dom)          # beam[4][n] es la fuerza cortante en cada punto
    flex = np.zeros_like(dom)           # beam[5][n] es el momento flector en cada punto
    slope = np.zeros_like(dom)          # beam[6][n] es la inclinacion en cada punto
    deflection = np.zeros_like(dom)     # beam[7][n] es la deflexion en cada punto
    beam = np.array([dom, loads, distloads, moment, shear, flex, slope, deflection])
    return beam

# Funcion para desactivar los botones de los apoyos

def update_disable(inp,txt,type,apoyo1,apoyo2,apo1button,apo2button):
    update(inp,txt)
    global beamtype, supp1, supp2
    if type.get() == 'Cantilever':
        apoyo1.config(state='disabled')
        apoyo2.config(state='disabled')
        apo1button.config(state='disabled')
        apo2button.config(state='disabled')
        beamtype = 1 #Viga empotrada
        supp1 = 0
        supp2 = 0
    else:
        apoyo1.config(state='normal')
        apoyo2.config(state='normal')
        apo1button.config(state='normal')
        apo2button.config(state='normal')
        beamtype = 2 #Viga apoyada

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

def point_load(pos,mag,cargas):
    global beam, scale
    x = int(float(pos.get())*scale)
    beam[1][x] -= float(mag.get())
    carga = 'Carga de '+mag.get()+' N en '+pos.get()+' m'
    cargas.insert(tk.END,carga)
    return beam

# Funcion que ubica los momentos

def point_moment(pos,mag,cargas):
    global beam, scale
    x = int(float(pos.get())*scale)
    beam[3][x] += float(mag.get())
    momento = 'Momento de '+mag.get()+' Nm en '+pos.get()+' m'
    cargas.insert(tk.END,momento)
    return beam

# Funcion que ubica las cargas distribuidas

def distributed_load(start, end, mag, cargas):
    global beam, scale
    xstart = int(float(start.get())*scale)
    xend = int(float(end.get())*scale)
    x = sp.symbols('x')
    equation = sp.sympify(mag.get())

    for i in range(0,xend-xstart+1):
        pos = i+xstart
        beam[2][pos] -= equation.subs(x,i*step)
    cargadis = 'Carga distribuida de '+mag.get()+' N/m entre '+start.get()+' m y '+end.get()+' m'
    cargas.insert(tk.END, cargadis)
    return beam
        

# Funcion que calcula las reacciones en los apoyos

def calculate_reactions():
    global beam, supp1, supp2, beamtype, R1, R2, MR, scale
    R1 = 0 #Reaccion en el apoyo 1
    R2 = 0 #Reaccion en el apoyo 2
    M1 = 0 #Momento en el apoyo 1
    M2 = 0 #Momento en el apoyo 2
    MR = 0 #Momento respecto al empotramiento

    if beamtype == 1:    #Si la viga es empotrada
        for i in range(len(beam[0])):
            MR -= beam[0][i] * (beam[1][i] + step * beam[2][i]) + beam[3][i]  #Sumatoria de momentos respecto al empotramiento
            R1 -= beam[1][i]+step*beam[2][i] #Sumatoria de cargas
        beam[1][0] = R1
        beam[3][0] = MR

    elif beamtype == 2: #Si la viga es apoyada
        for i in range(len(beam[0])):
            M1 += (beam[0][i]-supp1/scale)*(beam[1][i]+step*beam[2][i]) + beam[3][i]
            M2 += (beam[0][i]-supp2/scale)*(beam[1][i]+step*beam[2][i]) + beam[3][i]
            R1 = -M2/(supp1/scale-supp2/scale)
            R2 = -M1/(supp2/scale-supp1/scale)
        beam[1][supp1] += R1
        beam[1][supp2] += R2
       
    return R1, R2,  MR

# Funcion que grafica la viga

def plot_beam(beamgraph,figbeam):
    global beam, supp1, supp2, scale, beamtype
    figbeam.clear()
    ax = figbeam.add_subplot(111)
    if beamtype == 1:
        ax.plot([0,0],[5,-5],linewidth=5, color='gray', zorder=0)
        
    elif beamtype == 2:
        ax.plot([supp1/scale,supp1/scale],[0,-1],linewidth=3, color='red',zorder = 0)
        ax.plot([supp2/scale,supp2/scale],[0,-1],linewidth=3, color='red',zorder = 0)
        
    ax.plot(beam[0], np.zeros_like(beam[0]), linewidth=5, zorder=1)
    ax.set_ylim(-5.5,5.5)
    ax.yaxis.set_visible(False) 

    maxpoint = np.max(abs(beam[1]))
    maxdist = np.max(abs(beam[2]))
    truemax = max(maxpoint,maxdist)
    sizing  = 5/abs(truemax)

    for i in range(len(beam[0])):
        if beam[1][i] != 0:
            ratio = abs(beam[1][i]/truemax)
            xstart = i/scale

            if i == supp1 or i == supp2:
                ystart = -5*(ratio)-0.3
                ycomponent = 4*ratio
                color = 'r'
            else:
                ystart = 5*(ratio)+0.3
                ycomponent = -4*ratio
                color = 'k'

            xcomponent = 0
            
            width = ratio*0.025*len(beam[0])/scale
            height = ratio
            ax.arrow(xstart,ystart,xcomponent,ycomponent,head_width=width,head_length=height,linewidth=3, fc=color,ec=color)

    ax.plot(beam[0], -beam[2]*sizing, color='blue')
    ax.fill_between(beam[0], -beam[2]*sizing, 0, color='blue', alpha=0.5, interpolate=True)

    max_moment = beam[3][np.argmax(np.abs(beam[3]))]

    for i in range(len(beam[0])):
        if beam[3][i] != 0:
            ratio = abs(beam[3][i]/max_moment)
            x = i
            y = 0
            center = (x/scale, y)
            radiusy = ratio
            radiusx = ratio*0.05*len(beam[0])/scale

            if beam[3][i] > 0:
                theta1 = 270
                theta2 = 180
                arrow_start = (x/scale-radiusx, 0)
                arrow_end = (x/scale-radiusx, -0.5)
            else:
                theta1 = 0
                theta2 = 270
                arrow_start = (x/scale+radiusx, 0)
                arrow_end = (x/scale+radiusx, -0.5)

            if beamtype == 1 and i == 0:
                color = 'red'
            else:
                color = 'black'
            
            semicircle = patches.Arc(center, 2*radiusx, 2*radiusy, angle=0, theta1=theta1, theta2=theta2, edgecolor=color,zorder = 101)
            ax.add_patch(semicircle)

            arrow = patches.FancyArrowPatch(arrow_start, arrow_end, mutation_scale=15, color='red',zorder=102)
            ax.add_patch(arrow)


    plt.show()
    beamgraph.draw()

    # Funcion que grafica los cortantes 

def plot_shear(sheargraph,figshear):
    global beam, step, max_shear, max_shear_pos

    figshear.clear()
    ax = figshear.add_subplot(111)

    beam[4][0] = beam[1][0] + step*beam[2][0]

    for i in range(len(beam[0])-1):
        beam[4][i+1] = beam[4][i] + beam[1][i+1] + step*beam[2][i+1]

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
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    sheargraph.draw()

def plot_moment(momentgraph,figmoment):
    global beam, scale, step, max_moment, max_moment_pos, beamtype

    figmoment.clear()
    ax = figmoment.add_subplot(111)

    if beamtype == 1:
        beam[5][0] = -beam[3][0]

    if beamtype == 2:
        beam[5][0] = step*beam[4][0] - beam[3][0]

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
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    momentgraph.draw()

def add_youngs_modulus(inp):
    global E
    E = float(inp.get())

def add_inertia(inp):
    global I
    I = float(inp.get())

def add_neutral_axis_distance(inp):
    global c
    c = float(inp.get())

def add_thickness(inp):
    global t
    t = float(inp.get())

def add_first_moment_area(inp):
    global Q
    Q = float(inp.get())

def add_yield_strength(inp):
    global sy
    sy = float(inp.get())

def plot_slope_deflection(slopegraph,figslope,deflectiongraph,figdeflection):
    global beam, E, I, scale, step, beamtype, supp2, supp1

    figslope.clear()
    ax_slope = figslope.add_subplot(111)

    figdeflection.clear()
    ax_deflection = figdeflection.add_subplot(111)

    beam[6][0] = beam[5][0]/(E*I)
    for i in range(len(beam[0])-1):
        beam[6][i+1] = beam[6][i] + step*beam[5][i+1]/(E*I)
    
    beam[7][0] = beam[6][0]
    for i in range(len(beam[0])-1):
        beam[7][i+1] = beam[7][i] + step*beam[6][i+1]

    if beamtype == 1:
        c1 = -beam[6][0]
        c2 = -beam[7][0]
    
    elif beamtype == 2:
        c1 = (beam[7][supp2]-beam[7][supp1])/(supp1/scale-supp2/scale)
        c2 = -beam[7][supp1]/(E*I) - c1*supp1/scale

    for i in range(len(beam[0])):
        beam[6][i] += c1
        beam[7][i] += c1*beam[0][i] + c2

    max_slope = beam[6][np.argmax(np.abs(beam[6]))]
    max_slope_pos = beam[0][np.argmax(np.abs(beam[6]))]
    
    ax_slope.axhline(0, color='black', linewidth=1)
    ax_slope.plot([0,0],[0,beam[6][0]], color='green')
    ax_slope.plot(beam[0], beam[6], color='green')
    ax_slope.title.set_text('Inclinación de la viga')
    ax_slope.set_xlabel('Posicion (m)')
    ax_slope.set_ylabel('Inclinación (rad)')
    ax_slope.fill_between(beam[0], beam[6], 0, color='green', alpha=0.5)
    ax_slope.plot(max_slope_pos, max_slope, 'r*', markersize=10, label=f'Inclinación máxima: {max_slope} rad')
    ax_slope.legend()
    ax_slope.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    slopegraph.draw()

    max_deflection = beam[7][np.argmax(np.abs(beam[7]))]
    max_deflection_pos = beam[0][np.argmax(np.abs(beam[7]))]

    ax_deflection.axhline(0, color='black', linewidth=1)
    ax_deflection.plot([0,0],[0,beam[7][0]], color='purple')
    ax_deflection.plot(beam[0], beam[7], color='purple')
    ax_deflection.title.set_text('Deflexión de la viga')
    ax_deflection.set_xlabel('Posicion (m)')
    ax_deflection.set_ylabel('Deflexión (m)')
    ax_deflection.fill_between(beam[0], beam[7], 0, color='purple', alpha=0.5)
    ax_deflection.plot(max_deflection_pos, max_deflection, 'r*', markersize=10, label=f'Deflexión máxima: {max_deflection} m')
    ax_deflection.legend()
    ax_deflection.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    deflectiongraph.draw()

# Funcion para calcular el esfuerzo de von Mises

def von_mises_stress():
    global max_moment, max_moment_pos, max_shear, max_shear_pos, I, c, t, Q, sy, scale

    tau_xy1 = beam[4][max_moment_pos*scale]*Q/(I*t)
    sigma_x1 = max_moment*c/I
    sigma_y1 = 0
    vm1 = np.sqrt(sigma_x1**2 - sigma_x1*sigma_y1 + sigma_y1**2 + 3*tau_xy1**2)

    tau_xy2 = (max_shear*Q)/(I*t)
    sigma_x2 =beam[5][max_shear_pos*scale]*c/I
    sigma_y2 = 0
    vm2 = np.sqrt(sigma_x2**2 - sigma_x2*sigma_y2 + sigma_y2**2 + 3*tau_xy2**2)

    if vm1 > vm2:
        vmm = vm1
    else:
        vmm = vm2
    
    if vmm > sy:
        return(f'La viga falla con un esfuerzo de von Mises de {round(vmm),2} Pa')
    else:
        fs = sy/vmm
        return(f'La viga es segura con un factor de seguridad de {round(fs,2)}')


# Funcion que llama todas las funciones para calcular la viga

def calculate_beam(beamgraph,figbeam,sheargraph,figshear,momentgraph,
figmoment,slopegraph,figslope,deflectiongraph,figdeflection, results):
    global beam
    beam[4] = np.zeros_like(beam[0])
    beam[5] = np.zeros_like(beam[0])
    beam[6] = np.zeros_like(beam[0])
    beam[7] = np.zeros_like(beam[0])
    calculate_reactions()
    plot_beam(beamgraph,figbeam)
    plot_shear(sheargraph,figshear)
    plot_moment(momentgraph,figmoment)
    plot_slope_deflection(slopegraph,figslope,deflectiongraph,figdeflection)
    Factor_o_falla = von_mises_stress()
    results.config(text=Factor_o_falla)

def reset_all(entries,figures):
    global beam
    beam = build_beam(0)
    for entry in entries:
        entry.delete(0,'end')
    for figure in figures:
        figure.clear()
        figure.draw()