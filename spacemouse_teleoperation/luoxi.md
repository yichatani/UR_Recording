```

source ~/UR5_Policy/data_record_env/bin/activate
deactivate


cd spacemouse_teleoperation/
python 3DConnexion_UR5_Teleop_Gripper.py

cd ..

cd data_recording/
python record_data.py
```

<!-- train stage -->
cd /home/mainuser/UR5_Policy/Schwarz_DP3

bash /home/mainuser/UR5_Policy/Schwarz_DP3/scripts/train_policy.sh dp3 <task_name> 0888 0 0



