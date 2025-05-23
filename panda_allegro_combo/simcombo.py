import mujoco
import mujoco.viewer
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Simulation duration and timestep
sim_time = 10.0
timesteps = int(sim_time / model.opt.timestep)
time_array = np.linspace(0, sim_time, timesteps)

# Predefined joint trajectories
target_positions = np.zeros((timesteps, model.nu))  # model.nu = number of actuators

# Trajectories for joints
target_positions[:, 0] = 0.5 * time_array / sim_time                   
target_positions[:, 1] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)   
target_positions[:, 2] = -0.3 * time_array / sim_time                  
target_positions[:, 3] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)    
target_positions[:, 4] = 0.5 * time_array / sim_time                  
target_positions[:, 5] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)    
target_positions[:, 6] = -0.3 * time_array / sim_time                 
target_positions[:, 7] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)   
target_positions[:, 8] = 0.5 * time_array / sim_time                  
target_positions[:, 9] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)  
target_positions[:, 10] = -0.3 * time_array / sim_time                 
target_positions[:, 11] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)    
target_positions[:, 12] = 0.5 * time_array / sim_time                 
target_positions[:, 13] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)   
target_positions[:, 14] = -0.3 * time_array / sim_time                  
target_positions[:, 15] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)   
target_positions[:, 16] = 0.5 * time_array / sim_time                  
target_positions[:, 17] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)   
target_positions[:, 18] = -0.3 * time_array / sim_time                
target_positions[:, 19] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)   
target_positions[:, 20] = 0.3 * np.sin(2 * np.pi * 0.5 * time_array)  
target_positions[:, 21] = -0.3 * time_array / sim_time                  
target_positions[:, 22] = 0.2 * np.sin(2 * np.pi * 1.0 * time_array)   

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    i = 0
    while viewer.is_running() and i < timesteps:
        # Apply position targets to actuators
        data.ctrl[:] = target_positions[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        i += 1
        
