
import numpy as np


def update(inp,txt):
    newtxt = str(inp.get())
    txt.config(text=newtxt)

def validate_numeric_input(action, value_if_allowed):
    if action == '1':  
        if value_if_allowed.isdigit():
            return True
        else:
            return False
    else:
        return True
    
def build_beam(len):
    step = 0.01
    dom = np.arange(0, len, step)
    loads = np.zeros_like(dom)
    beam = np.array([dom, loads])
    return beam

def update_build(inp,txt):
    update(inp,txt)
    global beam
    beam = build_beam(float(inp.get()))

def update_supports(inp,txt,support):
    update(inp,txt)
    if support == 1:
        support1 = int(inp.get()*100)
    elif support == 2:
        support2 = int(inp.get()*100)
    print(support1,support2)
