---
title: Asimov Humanoid Playground
emoji: 🤖
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

Drive the Asimov v1 humanoid (23 DoF) in your browser. A motion-matching controller turns your keys into reference motion, and an RGMT tracking policy ([Robust and Generalized Humanoid Motion Tracking, arXiv:2601.23080](https://arxiv.org/abs/2601.23080)) tracks it on top of PD control. The policy runs with onnxruntime-web, physics runs in MuJoCo WebAssembly, and three.js renders the scene. The policy was trained in Newton, so what you see here is a sim2sim transfer.

Controls: **W/A/S/D** move, **Q/E** turn, **P** toggles the policy (off = pure PD on the reference, which tends to fall), **Space** resets.

Full source: [github.com/rsamf/asimov-rgmt](https://github.com/rsamf/asimov-rgmt)
