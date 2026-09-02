# pi0.5 pipelines

Two pipelines built on LeRobot's pi0.5 policy.

| Folder                                               | What it does                                                                                        | Start here                                            |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [`feeding_finetune/`](feeding_finetune/)             | LoRA fine-tune pi0.5 on the OmniGibson Kinova feeding task from NWB recordings                      | `bash examples/pi05/feeding_finetune/train.sh`        |
| [`libero_shared_autonomy/`](libero_shared_autonomy/) | Evaluate, drive and study shared-autonomy steering of pi0.5 on LIBERO with a SpaceMouse or keyboard | `./examples/pi05/libero_shared_autonomy/run.sh setup` |

The two do not share code: the feeding pipeline is a dataset converter plus a
`lerobot-train` launch; the LIBERO pipeline is a set of runners around
`lerobot.policies.pi05.steering`. Each folder's README is self-contained.
