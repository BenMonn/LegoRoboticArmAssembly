import mujoco
import mujoco.viewer
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Define key callback function
def key_callback(keycode):
    if chr(keycode) == 'W':
        data.ctrl[0] += 0.1 
    elif chr(keycode) == 'S':
        data.ctrl[0] -= 0.1  
    elif chr(keycode) == 'E':
        data.ctrl[1] += 0.1 
    elif chr(keycode) == 'D':
        data.ctrl[1] -= 0.1 
    elif chr(keycode) == 'R':
        data.ctrl[2] += 0.1 
    elif chr(keycode) == 'F':
        data.ctrl[2] -= 0.1 
    elif chr(keycode) == 'T':
        data.ctrl[3] += 0.1 
    elif chr(keycode) == 'G':
        data.ctrl[3] -= 0.1 
    elif chr(keycode) == 'Y':
        data.ctrl[4] += 0.1  
    elif chr(keycode) == 'H':
        data.ctrl[4] -= 0.1  
    elif chr(keycode) == 'U':
        data.ctrl[5] += 0.1 
    elif chr(keycode) == 'J':
        data.ctrl[5] -= 0.1  
    elif chr(keycode) == 'I':
        data.ctrl[6] += 0.1 
    elif chr(keycode) == 'K':
        data.ctrl[6] -= 0.1 
    elif chr(keycode) == 'O':
        data.ctrl[7] += 0.1  
    elif chr(keycode) == 'L':
        data.ctrl[7] -= 0.1 
    elif chr(keycode) == 'Q':
        data.ctrl[8] += 0.1 
    elif chr(keycode) == 'A':
        data.ctrl[8] -= 0.1 
    elif chr(keycode) == 'P':
        data.ctrl[9] += 0.1  
    elif chr(keycode) == 'Z':
        data.ctrl[9] -= 0.1 
    elif chr(keycode) == 'X':
        data.ctrl[10] += 0.1 
    elif chr(keycode) == 'C':
        data.ctrl[10] -= 0.1 
    elif chr(keycode) == 'V':
        data.ctrl[11] += 0.1 
    elif chr(keycode) == 'B':
        data.ctrl[11] -= 0.1 
    elif chr(keycode) == 'N':
        data.ctrl[12] += 0.1 
    elif chr(keycode) == 'M':
        data.ctrl[12] -= 0.1 
    elif chr(keycode) == '1':
        data.ctrl[13] += 0.1 
    elif chr(keycode) == '2':
        data.ctrl[13] -= 0.1 
    elif chr(keycode) == '3':
        data.ctrl[14] += 0.1 
    elif chr(keycode) == '4':
        data.ctrl[14] -= 0.1  
    elif chr(keycode) == '5':
        data.ctrl[15] += 0.1  
    elif chr(keycode) == '6':
        data.ctrl[15] -= 0.1 
    elif chr(keycode) == '7':
        data.ctrl[16] += 0.1  
    elif chr(keycode) == '8':
        data.ctrl[16] -= 0.1 
    elif chr(keycode) == '9':
        data.ctrl[17] += 0.1 
    elif chr(keycode) == '0':
        data.ctrl[17] -= 0.1
    elif chr(keycode) == '-':
        data.ctrl[18] += 0.1 
    elif chr(keycode0 == '=':
        data.ctrl[18] -= 0.1 
    elif chr(keycode) == '[':
        data.ctrl[19] += 0.1 
    elif chr(keycode) == ']':
        data.ctrl[19] -= 0.1 
    elif chr(keycode) == ';':
        data.ctrl[20] += 0.1
    elif chr(keycode) == ',':
        data.ctrl[20] -= 0.1
    elif chr(keycode) == '.':
        data.ctrl[21] += 0.1
    elif chr(keycode) == '/':
        data.ctrl[21] -= 0.1 
    elif chr(keycode) == '<':
        data.ctrl[22] += 0.1
    elif chr(keycode) == '>':
        data.ctrl[22] -= 0.1 

# Open interactive viewer with key callback
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


