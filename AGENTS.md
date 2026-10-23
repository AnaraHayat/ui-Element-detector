# AGENTS.md — UI Element Detector (read this first)

You are helping build a deep learning semester project in opencode. All project documents are in `docs/`. Read them in this order before doing anything:

1. `docs/rules.md` — rules you must follow
2. `docs/prd.md` — what we are building
3. `docs/architecture.md` — folders, stack, packages
4. `docs/phases.md` — the order of work
5. `docs/memory.md` — what is done and what is next
6. `docs/Design.md` — only for the frontend phase

## How to work
- Do only the current phase from `docs/memory.md`. Do one step at a time.
- Before coding, say in 2–3 lines what you will do. After coding, run it and show the result.
- After each finished step, update `docs/memory.md` (done, decisions, next) **and** `README.md` (what was done in that step, with numbers and commands).
- Use existing packages. Do not rebuild what a library already does.
- If something is unclear or conflicts with the docs, ask. Do not guess.
- Never delete or overwrite anything in `data/` or `results/` without asking.
- Keep answers short and plain.

## Key facts
- Dataset: RICO. Task: object detection on full screenshots (class + bounding box, YOLO format), 15 classes. Models to compare (5–7, user's list): DenseNet, Xception, PyramidNet, Residual Attention Network, SENet, latest YOLO. The "published after 2022" filter was dropped by the user.
- No image generation of any kind (no synthetic or copy-paste images).
- Pipeline: Preprocessing → Split 70:20:10 (train:test:validate) → Training → Testing → Evaluation.
