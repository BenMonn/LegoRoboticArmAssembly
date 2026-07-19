After downloading all the code, follow the instructions below to run the code:

Instructions:
1. Run `python training/train.py` until you obtain 100% success consistently
2. Next run `python training/hold_train.py` from your best checkpoint from train.py until you reach 100% success consistently
3. Then run `python training/grasp_train.py` from your best hold_train.py checkpoint. This file is still a work in progress, so results will vary
4. Lastly run `python training/hypernet_goal_train.py` from your best checkpoint from grasp_train.py to run the hypernetwork
