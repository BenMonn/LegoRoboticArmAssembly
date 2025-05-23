import mujoco
import mujoco.viewer
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set initial control values (e.g., all zeros)
    data.ctrl[:] = 0.5  

    # Run interactive loop
    while viewer.is_running():
        t = data.time

        data.ctrl[0] = 0.3 * np.sin(t) 
        data.ctrl[1] = 0.2 * np.sin(t)  
        data.ctrl[2] = 0.1 * np.sin(t)  
        data.ctrl[3] = 0.05 * np.sin(t) 
        data.ctrl[5] = 0.3 * np.sin(t)  
        data.ctrl[6] = 0.2 * np.sin(t)  
        data.ctrl[7] = 0.1 * np.sin(t)  
        data.ctrl[8] = 0.05 * np.sin(t) 
        data.ctrl[9] = 0.3 * np.sin(t)   
        data.ctrl[10] = 0.2 * np.sin(t)  
        data.ctrl[11] = 0.1 * np.sin(t)  
        data.ctrl[12] = 0.05 * np.sin(t) 
        data.ctrl[13] = 0.2 * np.sin(t)
        data.ctrl[14] = 0.1 * np.sin(t)  
        data.ctrl[15] = 0.05 * np.sin(t) 
        data.ctrl[16] = 0.3 * np.sin(t)  
        data.ctrl[17] = 0.2 * np.sin(t)  
        data.ctrl[18] = 0.1 * np.sin(t)   
        data.ctrl[19] = 0.05 * np.sin(t)  
        data.ctrl[20] = 0.2 * np.sin(t)   
        data.ctrl[21] = 0.1 * np.sin(t)   
        data.ctrl[22] = 0.05 * np.sin(t)  

        mujoco.mj_step(model, data)
        viewer.sync()


