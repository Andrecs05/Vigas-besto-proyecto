
import numpy as np
import re


def update(inp,txt):
    newtxt = str(inp.get())
    txt.config(text=newtxt)

def validate_numeric_input(action, value_if_allowed):
    if action == '1': 
        if re.match(r'^\d*\.?\d*$', value_if_allowed):
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
    global supp1, supp2
    if support == 1:
        supp1 = int(float(inp.get())*100)
    elif support == 2:
        supp2 = int(float(inp.get())*100)
    if 'supp1' in globals():
        print(f"supp1: {supp1}")
    else:
        print("supp1 is not assigned a value.")
    
    if 'supp2' in globals():
        print(f"supp2: {supp2}")
    else:
        print("supp2 is not assigned a value.")
    print(beam[0][supp1])
