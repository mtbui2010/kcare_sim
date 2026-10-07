# Learn: from simulation to a real robot

Seven short chapters on the ideas kcare-sim is built on. Each uses this repository's code and
numbers as the example.

| # | chapter | question it answers |
|---|---|---|
| 1 | [Running the robot's own software in a simulator](01-real-software.md) | how can unmodified robot code drive a simulated robot? |
| 2 | [A scene from a floor plan and photos](02-scene-from-photos.md) | how close to the real home does the scene need to be, and how do you get there? |
| 3 | [Making renders look like the cameras](03-camera-model.md) | why does a perfect render hurt a policy, and what do you add? |
| 4 | [Depth sensors are not perfect](04-depth.md) | what does a real depth image look like, and why does it matter? |
| 5 | [Demonstrations from a scripted expert](05-expert-and-randomisation.md) | where do 500 training episodes come from in two hours? |
| 6 | [VLA policies and action chunks](06-vla.md) | what does the policy take in and put out, and how does it run at 30 Hz? |
| 7 | [Evaluating without fooling yourself](07-evaluation.md) | is 52 % better than 8 %? is 70 % better than 68 %? |

## Glossary

VLA (vision-language-action model)
:   a neural network that takes camera images, the robot's joint state and a text instruction, and
    outputs robot actions. Here: SmolVLA.

Action chunk
:   a block of future actions (50 joint targets here) predicted at once and executed over time.

Domain randomisation
:   varying everything that should not matter (light, colours, poses, camera settings) in the
    training data so the policy learns to ignore it.

Sim-to-real gap
:   whatever differs between simulation and reality and makes a policy that works in one fail in
    the other: looks, sensor noise, physics, timing.

ENV entry
:   kcare's description of a place the robot can visit: where to stand, which way to face, the
    surface height.

Spot
:   a surface in the house layout (a desk, a counter) that becomes an ENV entry.
