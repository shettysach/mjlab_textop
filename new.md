1. ONNX cpu hack was becaus it used to crash on CUDA. Let it stay.
2. Remove the configuration to have arbitrary future_steps. Let it be the constant `FUTURE_STEPS` = 5.
3. Yes this is a major issue to address. Ensure they close properly.
4. I'm sure TextOpRobotMDAR produces proper 50 fps. Could we remove it?
5. We'll deal with this later. We need TextOp compatibility.

Regarding eliminating code. I agree with all except `notes/ONNX_CUDA.md`.

Regarding refactors,
- I agree with task configuration part. We can consolidate much better.
- Totally agree with refactoring OnlineMotionCommand.
