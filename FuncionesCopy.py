
import numpy as np
import re
import matplotlib.pyplot as plt
import sympy as sp
import tkinter as tk
from tkinter import ttk
import math
import matplotlib.patches as patches

# Funcion para actualizar el label txt con el texto de inp
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

# Funcion para validar que la entrada sea numerica con varios decimales
def validate_numeric_input_more_decimals(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^-?\d*\.?\d*$', value_if_allowed):
            return True
        else:
            return False    
    else:
        return True
    
# Funcion para construir la viga y las cargas en una matrix 2xn donde n es la cantidad de puntos en la viga
def build_beam(inp,datos):
    global scale, step, beam
    # Definir la longitud de la viga y el paso
    len = float(inp.get())              
    order = math.floor(math.log10(len))
    step = 10 ** (order-4)
    scale = 1/step
    
    # Crear la viga como un diccionario con las propiedades de la viga
    beam = {}
    beam['x_coordinate'] = np.arange(0, len+step, step) 
    beam['point_loads'] = np.zeros_like(beam['x_coordinate'])         
    beam['distributed_loads'] = np.zeros_like(beam['x_coordinate'])      
    beam['point_moment'] = np.zeros_like(beam['x_coordinate'])        
    beam['shear'] = np.zeros_like(beam['x_coordinate'])        
    beam['bending_moment'] = np.zeros_like(beam['x_coordinate'])           
    beam['slope'] = np.zeros_like(beam['x_coordinate'])          
    beam['deflection'] = np.zeros_like(beam['x_coordinate'])     

    # Introducir la longitud de la viga en el cuadro de datos
    longitud = 'Longitud de la viga: '+str(len)+' m'
    datos.insert(tk.END,longitud)       

# Funcion para definir el tipo de viga y desactivar los botones de los apoyos si es empotrada
def type_disable(type,apoyo1,apoyo2,apo1button,apo2button,datos):
    global beamtype, supp1, supp2
    if type.get() == 'Cantilever':
        apoyo1.config(state='disabled')
        apoyo2.config(state='disabled')
        apo1button.config(state='disabled')
        apo2button.config(state='disabled')
        beamtype = 1 #Viga empotrada
        supp1 = 0
        supp2 = 0
        type = 'Viga empotrada'
    else:
        apoyo1.config(state='normal')
        apoyo2.config(state='normal')
        apo1button.config(state='normal')
        apo2button.config(state='normal')
        beamtype = 2 #Viga apoyada
        type = 'Viga apoyada'

    datos.insert(tk.END,type)       # Introducir el tipo de viga en el cuadro de datos

# Funcion que llama a la funcion update y ubicar los apoyos
def update_supports(inp,support,datos): #support = 1 para apoyo 1, support = 2 para apoyo 2
    global supp1, supp2, scale
    if support == 1:
        supp1 = int(float(inp.get())*scale) 
        support_text = 'Apoyo 1 en '+inp.get()+' m'

    elif support == 2:
        supp2 = int(float(inp.get())*scale) 
        support_text = 'Apoyo 2 en '+inp.get()+' m'

    datos.insert(tk.END,support_text)    # Introducir la ubicacion de los apoyos en el cuadro de datos

# Funcion para ubicar las cargas puntuales
def point_load(pos,mag,datos):
    global beam, scale

    # Ubicar la carga ingresada por el usuario en la viga
    x = int(float(pos.get())*scale) 
    beam['point_loads'][x] -= float(mag.get())

    # Limpiar las entradas de las cargas puntuales
    pos.delete(0,'end')    
    mag.delete(0,'end')

    # Introducir la carga puntual en el cuadro de datos
    carga = 'Carga de '+str(beam['point_loads'][x])+' N en '+str(x*step)+' m'
    datos.insert(tk.END,carga)
    return beam

# Funcion para ubicar los momentos puntuales
def point_moment(pos,mag,datos):
    global beam, scale

    # Ubicar el momento ingresado por el usuario en la viga
    x = int(float(pos.get())*scale)
    beam['point_moment'][x] += float(mag.get())

    # Limpiar las entradas de los momentos puntuales
    pos.delete(0,'end')
    mag.delete(0,'end')

    # Introducir el momento puntual en el cuadro de datos
    momento = 'Momento de '+str(beam['point_moment'][x])+' Nm en '+str(x*step)+' m'
    datos.insert(tk.END,momento)
    return beam

# Funcion que ubica las cargas distribuidas
def distributed_load(start, end, mag, datos):
    global beam, scale

    # Definir los puntos de inicio y fin de la carga distribuida y su expresión
    xstart = int(float(start.get())*scale)
    xend = int(float(end.get())*scale)
    x = sp.symbols('x')
    equation = sp.sympify(mag.get())

    # Evaluar la expresión en cada punto del intervalo y sumarla a la viga
    for i in range(0,xend-xstart+1):
        pos = i+xstart
        beam['distributed_loads'][pos] -= equation.subs(x,i*step)

    # Limpiar las entradas de las cargas distribuidas
    start.delete(0,'end')
    end.delete(0,'end')
    mag.delete(0,'end')

    # Introducir la carga distribuida en el cuadro de datos
    cargadis = 'Carga distribuida de '+str(equation)+' N/m entre '+str(xstart*step)+' m y '+str(xend*step)+' m'
    datos.insert(tk.END, cargadis)
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
        # Se calcula el momento respecto al empotramiento y 
        # la sumatoria de cargas para definir las reacciones
        for i in range(len(beam['x_coordinate'])):
            MR -= beam['x_coordinate'][i] * (beam['point_loads'][i] + step * beam['distributed_loads'][i]) + beam['point_moment'][i]  
            R1 -= beam['point_loads'][i]+step*beam['distributed_loads'][i] 
        beam['point_loads'][0] = R1
        beam['point_moment'][0] = MR

    elif beamtype == 2: #Si la viga es apoyada
        # Se calculan los momentos en los apoyos y se 
        # despejan las reacciones de la sumatoria de momentos
        for i in range(len(beam['x_coordinate'])):
            M1 += (beam['x_coordinate'][i]-supp1/scale)*(beam['point_loads'][i]+step*beam['distributed_loads'][i]) + beam['point_moment'][i]
            M2 += (beam['x_coordinate'][i]-supp2/scale)*(beam['point_loads'][i]+step*beam['distributed_loads'][i]) + beam['point_moment'][i]
            R1 = -M2/(supp1/scale-supp2/scale)
            R2 = -M1/(supp2/scale-supp1/scale)
        beam['point_loads'][supp1] += R1
        beam['point_loads'][supp2] += R2
       
    return R1, R2,  MR

# Funcion que grafica la viga
def plot_beam(beamgraph,figbeam):
    global beam, supp1, supp2, scale, beamtype
    figbeam.clear()
    ax = figbeam.add_subplot(111)
    if beamtype == 1:
        ax.plot([0,0],[5,-5],linewidth=5, color='gray', label=f'Reacción de {R1} N', zorder=0)
        ax.plot([0,0],[5,-5],linewidth=5, color='gray', label=f'Reacción de {MR} Nm', zorder=0)
        
    elif beamtype == 2:
        ax.plot([supp1/scale,supp1/scale],[0,-1],linewidth=3, color='red', label=f'Reacción de {R1} N', zorder = 0)
        ax.plot([supp2/scale,supp2/scale],[0,-1],linewidth=3, color='orange', label=f'Reacción de {R2} N', zorder = 0)
        ax.legend()
        
    ax.plot(beam['x_coordinate'], np.zeros_like(beam['x_coordinate']), linewidth=5, zorder=1)
    ax.set_ylim(-1,5.5)
    ax.yaxis.set_visible(False) 

    maxpoint = np.min(beam['point_loads'])
    maxdist = np.min(beam['distributed_loads'])
    truemax = min(maxpoint,maxdist)
    sizing  = 5/abs(truemax)

    for i in range(len(beam['x_coordinate'])):
        if beam['point_loads'][i] != 0 and i != supp1 and i != supp2:
            ratio = abs(beam['point_loads'][i]/truemax)
            xstart = i/scale

            ystart = 5*(ratio)+0.3
            ycomponent = -4*ratio
            color = 'k'
            xcomponent = 0
            
            width = ratio*0.025*len(beam['x_coordinate'])/scale
            height = ratio
            ax.arrow(xstart,ystart,xcomponent,ycomponent,head_width=width,head_length=height,linewidth=3, fc=color,ec=color)

    ax.plot(beam['x_coordinate'], -beam['distributed_loads']*sizing, color='blue')
    ax.fill_between(beam['x_coordinate'], -beam['distributed_loads']*sizing, 0, color='blue', alpha=0.5, interpolate=True)

    max_moment = beam['point_moment'][np.argmax(np.abs(beam['point_moment']))]

    for i in range(len(beam['x_coordinate'])):
        if beam['point_moment'][i] != 0:
            ratio = abs(beam['point_moment'][i]/max_moment)
            x = i
            y = 0
            center = (x/scale, y)
            radiusy = ratio
            radiusx = ratio*0.05*len(beam['x_coordinate'])/scale

            if beam['point_moment'][i] > 0:
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

# Funcion que grafica los cortantes
def plot_shear(sheargraph,figshear):
    global beam, step, max_shear, max_shear_pos

    # Se limpia la grafica de cortantes
    figshear.clear()
    ax = figshear.add_subplot(111)

    # Se define el cortante en el primer punto de la viga
    beam['shear'][0] = beam['point_loads'][0] + step*beam['distributed_loads'][0]

    # Se calcula el cortante integrando las cargas mediante el método de Euler
    for i in range(len(beam['x_coordinate'])-1):
        beam['shear'][i+1] = beam['shear'][i] + beam['point_loads'][i+1] + step*beam['distributed_loads'][i+1]

    # Se calcula la fuerza cortante máxima y su posición
    max_shear = beam['shear'][np.argmax(np.abs(beam['shear']))]
    max_shear_pos = beam['x_coordinate'][np.argmax(np.abs(beam['shear']))]

    # Se grafica el cortante y el máximo
    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam['shear'][0]],color='blue')
    ax.plot(beam['x_coordinate'], beam['shear'],color='blue')
    ax.title.set_text('Fuerzas cortantes')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Fuerza cortante (N)')
    ax.fill_between(beam['x_coordinate'], beam['shear'], 0, color='blue', alpha=0.5, interpolate=True) 
    ax.plot(max_shear_pos, max_shear, 'r*', markersize=10, label=f'Fuerza cortante máxima: {round(max_shear,2)} N')
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    sheargraph.draw()   # Actualiza la interfaz grafica

# Funcion que grafica los momentos flectores
def plot_moment(momentgraph,figmoment):
    global beam, scale, step, max_moment, max_moment_pos, beamtype

    # Se limpia la grafica de momentos flectores
    figmoment.clear()
    ax = figmoment.add_subplot(111)

    # Se define el momento flector en el primer punto de la viga
    # segun el tipo de viga
    if beamtype == 1:
        beam['bending_moment'][0] = -beam['point_moment'][0]

    if beamtype == 2:
        beam['bending_moment'][0] = step*beam['shear'][0] - beam['point_moment'][0]

    # Se calcula el momento flector integrando los cortantes mediante el método de Euler
    for i in range(len(beam['x_coordinate'])-1):
        beam['bending_moment'][i+1] = beam['bending_moment'][i] + step*beam['shear'][i+1] - beam['point_moment'][i+1]

    # Se calcula el momento flector máximo y su posición
    max_moment = beam['bending_moment'][np.argmax(np.abs(beam['bending_moment']))]
    max_moment_pos = beam['x_coordinate'][np.argmax(np.abs(beam['bending_moment']))]

    # Se grafica el momento flector y el máximo
    ax.axhline(0, color='black', linewidth=1)
    ax.plot([0,0],[0,beam['bending_moment'][0]], color='orange')
    ax.plot(beam['x_coordinate'], beam['bending_moment'], color='orange')
    ax.title.set_text('Momento flector')
    ax.set_xlabel('Posicion (m)')
    ax.set_ylabel('Momento flector (Nm)')
    ax.fill_between(beam['x_coordinate'], beam['bending_moment'], 0, color='orange', alpha=0.5) 
    ax.plot(max_moment_pos, max_moment, 'r*', markersize=10, label=f'Momento flexor máximo: {round(max_moment,2)} Nm')
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    momentgraph.draw()  # Actualiza la interfaz grafica

# Funcion que define el modulo de Young
def add_youngs_modulus(inp,datos):
    global E
    E = float(inp.get())*10**6
    young = 'Módulo de Young: '+inp.get()+' MPa'
    datos.insert(tk.END,young)

# Funcion que define la inercia de la seccion transversal
def add_inertia(inp,datos):
    global I
    I = float(inp.get())*10**-12
    inertia = 'Inercia de la sección transversal: '+inp.get()+' m^4'
    datos.insert(tk.END,inertia)

# Funcion que define la distancia al eje neutro
def add_neutral_axis_distance(inp,datos):
    global c
    c = float(inp.get())
    distancia = 'Distancia al eje neutro: '+inp.get()+' m'
    datos.insert(tk.END,distancia)

# Funcion que define el espesor de la seccion transversal
def add_thickness(inp,datos):
    global t
    t = float(inp.get())
    espesor = 'Espesor de la sección transversal: '+inp.get()+' m'
    datos.insert(tk.END,espesor)

# Funcion que define el primer momento de area
def add_first_moment_area(inp,datos):
    global Q
    Q = float(inp.get())*10**-9
    momento1 = 'Primer momento de área: '+inp.get()+' mm^3'
    datos.insert(tk.END,momento1)

# Funcion que define la resistencia a la fluencia
def add_yield_strength(inp,datos):
    global sy
    sy = float(inp.get())*10**6
    resistencia = 'Resistencia a la fluencia: '+inp.get()+' MPa'
    datos.insert(tk.END,resistencia)

# Funcion que grafica la inclinacion y la deflexion
def plot_slope_deflection(slopegraph,figslope,deflectiongraph,figdeflection):
    global beam, E, I, scale, step, beamtype, supp2, supp1

    # Se limpian las graficas de inclinacion y deflexion
    figslope.clear()
    ax_slope = figslope.add_subplot(111)
    figdeflection.clear()
    ax_deflection = figdeflection.add_subplot(111)

    # Se calcula la integral y doble integral del momento en el primer punto de la viga
    beam['slope'][0] = beam['bending_moment'][0]/(E*I)
    beam['deflection'][0] = beam['slope'][0]

    # Se calcula ambas integrales mediante el método de Euler
    for i in range(len(beam['x_coordinate'])-1):
        beam['slope'][i+1] = beam['slope'][i] + step*beam['bending_moment'][i+1]/(E*I)
    
    for i in range(len(beam['x_coordinate'])-1):
        beam['deflection'][i+1] = beam['deflection'][i] + step*beam['slope'][i+1]

    # Se calculan las constantes de integracion para la inclinacion y la deflexion
    # segun el tipo de viga
    if beamtype == 1:
        c1 = -beam['slope'][0]
        c2 = -beam['deflection'][0]
    
    elif beamtype == 2:
        c1 = (beam['deflection'][supp2]-beam['deflection'][supp1])/(supp1/scale-supp2/scale)
        c2 = -beam['deflection'][supp1]/(E*I) - c1*supp1/scale

    # Se suman las constantes de integracion para obtener la inclinacion y la deflexion
    for i in range(len(beam['x_coordinate'])):
        beam['slope'][i] += c1
        beam['deflection'][i] += c1*beam['x_coordinate'][i] + c2

    # Se calcula la inclinacion y la deflexion maxima y su posicion
    max_slope = beam['slope'][np.argmax(np.abs(beam['slope']))]
    max_slope_pos = beam['x_coordinate'][np.argmax(np.abs(beam['slope']))]
    max_deflection = beam['deflection'][np.argmax(np.abs(beam['deflection']))]
    max_deflection_pos = beam['x_coordinate'][np.argmax(np.abs(beam['deflection']))]
    
    # Se grafican la inclinacion y la deflexion y sus maximos
    ax_slope.axhline(0, color='black', linewidth=1)
    ax_slope.plot([0,0],[0,beam['slope'][0]], color='green')
    ax_slope.plot(beam['x_coordinate'], beam['slope'], color='green')
    ax_slope.title.set_text('Inclinación de la viga')
    ax_slope.set_xlabel('Posicion (m)')
    ax_slope.set_ylabel('Inclinación (rad)')
    ax_slope.fill_between(beam['x_coordinate'], beam['slope'], 0, color='green', alpha=0.5)
    ax_slope.plot(max_slope_pos, max_slope, 'r*', markersize=10, label=f'Inclinación máxima: {np.format_float_scientific(max_slope,precision = 2)} rad')
    ax_slope.legend()
    ax_slope.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    slopegraph.draw()   # Actualiza la interfaz grafica

    ax_deflection.axhline(0, color='black', linewidth=1)
    ax_deflection.plot([0,0],[0,beam['deflection'][0]], color='purple')
    ax_deflection.plot(beam['x_coordinate'], beam['deflection'], color='purple')
    ax_deflection.title.set_text('Deflexión de la viga')
    ax_deflection.set_xlabel('Posicion (m)')
    ax_deflection.set_ylabel('Deflexión (m)')
    ax_deflection.fill_between(beam['x_coordinate'], beam['deflection'], 0, color='purple', alpha=0.5)
    ax_deflection.plot(max_deflection_pos, max_deflection, 'r*', markersize=10, label=f'Deflexión máxima: {np.format_float_scientific(max_deflection, precision = 2)    } m')
    ax_deflection.legend()
    ax_deflection.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    deflectiongraph.draw()  # Actualiza la interfaz grafica
 
# Funcion que retorna el esfuerzo de von Mises y el factor de seguridad
def von_mises_stress():
    global max_moment, max_moment_pos, max_shear, max_shear_pos, I, c, t, Q, sy, scale

    # Se calcula el esfuerzo de von Mises en el punto de mayor momento flector y cortante
    tau_xy1 = beam['shear'][int(max_moment_pos*scale)]*Q/(I*t)
    sigma_x1 = max_moment*c/I
    sigma_y1 = 0
    vm1 = np.sqrt(sigma_x1**2 - sigma_x1*sigma_y1 + sigma_y1**2 + 3*tau_xy1**2)

    tau_xy2 = (max_shear*Q)/(I*t)
    sigma_x2 =beam['bending_moment'][int(max_shear_pos*scale)]*c/I
    sigma_y2 = 0
    vm2 = np.sqrt(sigma_x2**2 - sigma_x2*sigma_y2 + sigma_y2**2 + 3*tau_xy2**2)

    # Se calcula el esfuerzo de von Mises maximo y el factor de seguridad
    if vm1 > vm2:
        vmm = vm1
    else:
        vmm = vm2
    
    if vmm > sy:
        return(f'La viga falla con un esfuerzo de von Mises de {round(vmm*10**-6,2)} MPa')
    else:
        fs = sy/vmm
        return(f'La viga es segura con un factor de seguridad de {round(fs,2)}')

# Funcion que llama todas las funciones para calcular la viga
def calculate_beam(beamgraph,figbeam,sheargraph,figshear,momentgraph,
figmoment,slopegraph,figslope,deflectiongraph,figdeflection, results):
    global beam

    # Se reinician las matrices de la viga
    beam['shear'] = np.zeros_like(beam['x_coordinate'])
    beam['bending_moment'] = np.zeros_like(beam['x_coordinate'])
    beam['slope'] = np.zeros_like(beam['x_coordinate'])
    beam['deflection'] = np.zeros_like(beam['x_coordinate'])

    calculate_reactions()
    plot_beam(beamgraph,figbeam)
    plot_shear(sheargraph,figshear)
    plot_moment(momentgraph,figmoment)
    plot_slope_deflection(slopegraph,figslope,deflectiongraph,figdeflection)
    Factor_o_falla = von_mises_stress()
    results.config(text=Factor_o_falla)     # Actualiza el label de resultado
