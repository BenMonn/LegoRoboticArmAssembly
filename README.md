# LegoRoboticArmAssembly
Building lego sets with a robotic arm in mujoco simulation.

The Franka Emika Panda robot arm and the Allegro right hand are files taken from the mujoco_menagerie github. Please check out their work as well, as they are doing great things. You can find a link to their github below that also contains the original source files for the Franka Emika Panda robot arm and the Allegro right hand.

https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md

Instructions:
1. Download code and run `python training/train.py` until you obtain 100% success consistently
2. Next run `python training/hold_train.py` from your best checkpoint from train.py until you reach 100% success consistently
3. Then run `python training/grasp_train.py` from your best hold_train.py checkpoint. This file is still a work in progress, so results will vary
4. Lastly run `python training/hypernet_goal_train.py` from your best checkpoint from grasp_train.py to run the hypernetwork
