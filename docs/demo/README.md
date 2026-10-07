# Demo: train and run — step by step

Open **`kaair_demo.ipynb`** in VS Code web (code-server on the A6000), choose the kernel
**"kaair demo (host)"** (top right), then run the cells in order with `Shift+Enter`.
`kaair_demo_executed.ipynb` is the same notebook already run once, so every image, video and plot
is visible without running anything.

| Step | What it shows | Time |
|---|---|---|
| 0 | environment check (container, GPUs, NAS, trained checkpoints) | 10 s |
| 1 | the robot in MuJoCo at the open-drawer ready pose | 1 min |
| 2 | real vs simulated camera images | instant (1 min to re-render) |
| 3 | the scripted expert opening a chosen drawer (`DRAWER`, `SEED`) | 1 min |
| 4 | a 6-episode dataset, and what one frame contains | 3–4 min |
| 5 | 200 training steps: the loss going down (next to the real run's loss curve) | 3–5 min |
| 6 | the trained VLA with any sentence (`PROMPT`, `SEED`) | 1–2 min |
| 7 | success per checkpoint and per drawer, sample success / failure clips | 10 s |
| — | clean-up: delete the mini dataset and checkpoint from the NAS | 1 s |

How it works:

- The kernel runs on the host (`docs/demo/.venv`: ipykernel, matplotlib, pandas). Every
  simulation / ML command goes into the `kaair_sim_humble` container through `docker exec`
  (`demo_lib.sh`). Results are read back through the host paths of `/sim` and `/nas`.
- Videos come from `sim/scripts/demo.py` (`expert` / `run`): an overview camera plus the two images
  the policy receives, with the sentence and how far each drawer is open.
- Only GPUs 0 and 3 are used (`demo_lib.pick_gpu`). Nothing is sent to the real robot.

Set-up, done once (already done on the A6000):

```bash
code-server --install-extension ms-toolsai.jupyter
cd ~/trung_workdir/kaair_sim/docs/demo
python3 -m venv .venv && .venv/bin/pip install ipykernel matplotlib pandas pyyaml nbclient
.venv/bin/python -m ipykernel install --user --name kaair-demo --display-name "kaair demo (host)"
```
