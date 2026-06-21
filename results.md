Task: Mjlab-TextOp-Flat-Unitree-G1
Anchor: pelvis
Motion: TextOp walking NPZ
Training: 4096 envs, 2000 iterations
Eval: 1024 envs, corruption enabled
Result:
  success_rate: 0.9795
  mpkpe: 0.2420
  r_mpkpe: 0.0730
  joint_vel_error: 0.8067
  ee_pos_error: 0.0845
  ee_ori_error: 0.1584
Qualitative: robot walks slightly ahead of target but stable.
