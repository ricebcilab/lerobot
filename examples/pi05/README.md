# pi0.5 pipelines

Two pipelines built on LeRobot's pi0.5 policy.

| Folder                                   | What it does                                                                               | Start here                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| [`feeding_finetune/`](feeding_finetune/) | LoRA fine-tune pi0.5 on the OmniGibson Kinova feeding task from NWB recordings             | `bash examples/pi05/feeding_finetune/train.sh` |
| [`mars/`](mars/)                         | MARS shared autonomy: a cursor task with a human study and a LIBERO example steering pi0.5 | `examples/pi05/mars/README.md`                 |

The two do not share code: the feeding pipeline is a dataset converter plus a
`lerobot-train` launch; the LIBERO pipeline is a set of runners around
`lerobot.policies.pi05.steering`. Each folder's README is self-contained.

`mars/` is the MARS project: shared-autonomy steering of generative policies by a person's
low-dimensional command. `mars/cursor/` runs it on a 2D cursor task with a flow-matching policy, a
human-study portal, a simulated users x tasks grid and the paper's analysis; `mars/libero/` runs the
same steering methods on pi0.5 in LIBERO. Start with [`mars/README.md`](mars/README.md).
