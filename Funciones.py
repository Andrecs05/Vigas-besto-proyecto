
import numpy as np
import re

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

def build_beam(len):
    step = 0.01
    dom = np.arange(0, len, step)
    loads = np.zeros_like(dom)
    moment = np.zeros_like(dom)
    beam = np.array([dom, loads,moment])
    return beam

# Funcion para desactivar los botones de los apoyos

def update_disable(inp,txt,type,apoyo1,apoyo2,apo1button,apo2button):
    update(inp,txt)
    if type.get() == 'Cantilever':
        apoyo1.config(state='disabled')
        apoyo2.config(state='disabled')
        apo1button.config(state='disabled')
        apo2button.config(state='disabled')
    else:
        apoyo1.config(state='normal')
        apoyo2.config(state='normal')
        apo1button.config(state='normal')
        apo2button.config(state='normal')

# Funcion que llama a la funcion build_beam y update

def update_build(inp,txt):
    update(inp,txt)
    global beam
    beam = build_beam(float(inp.get()))

# Funcion que llama a la funcion update y ubica los soportes

def update_supports(inp,txt,support):
    update(inp,txt)
    global supp1, supp2
    if support == 1:
        supp1 = int(float(inp.get())*100)
    elif support == 2:
        supp2 = int(float(inp.get())*100)


