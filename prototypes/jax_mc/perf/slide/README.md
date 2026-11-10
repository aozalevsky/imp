# A slide of the node-against-node speeds

`slide.png` shows what `../node_results.jsonl` holds (see `../README.md`,
"One node against one node"), for the NPC basket and the mature NPC: the
GPU node's speed over the CPU node's, and four speeds on a log axis (one
CPU thread, all 64 threads, one GPU, all four). The numbers are
preliminary.

- `build.py` writes `slide.pptx`: one 1920 x 1080 pt slide of native text
  boxes, shapes and pictures, so it can be edited and copied into a deck
  (PowerPoint or Keynote). It takes the speeds from the records, the texts
  from `texts.json` and the pictures from `models.json`.
- `models.py` makes the two pictures and `models.json` from the deposited
  models: the mature snapshot of Latham et al., PNAS 2025 (model 6 of
  `NPC_assembly.cif` in PM_NPC_Assembly) and the 8-fold basket model of
  Singh, Soni, et al., Cell 2024 (`yeast_npcbasket_c8.cif` in NPC_Basket).

What the slide says, and how to read it:

- **"N x more sampling per node"** is the MC steps/s of the four GPUs over
  that of the 64 CPU threads, rounded to two digits (2,761 and 1,256).
  The steps are summed over all chains that run in parallel: 32,768 and
  4,096 on the GPU node, 64 on the CPU node. "1 month on the CPU node is
  about N minutes on the GPU node" is 30 days divided by that factor.
- **The C++ side is the fastest one measured:** the fast build with a
  process on every hardware thread.
- **Scale bar and mass.** Both pictures are drawn at one scale, so the two
  500 A bars have the same length. The bar holds for horizontal distances;
  the views are tilted. The mass is that of the model's chains at full
  length, from their sequences: 36.3 MDa for the mature NPC snapshot
  (the scaffold nucleoporins, not the whole NPC) and 15.6 MDa for the
  basket model with all 8 spokes (basket proteins and the nuclear ring's
  Y-complexes). The basket protocol samples one spoke, with symmetric
  copies: 39 rigid bodies and 309 flexible beads.

Rebuild:

```
python models.py NPC_assembly.cif yeast_npcbasket_c8.cif   # pictures, models.json
python build.py                                            # slide.pptx
soffice --headless --convert-to pdf slide.pptx
pdftoppm -png -singlefile -scale-to-x 3840 -scale-to-y 2160 slide.pdf slide
```

`models.py` needs numpy, matplotlib and Pillow, `build.py` python-pptx.
`slide.png` here is that render (LibreOffice); `slide.pptx` and `slide.pdf`
are not kept in the repository.
