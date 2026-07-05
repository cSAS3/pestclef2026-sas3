# Final selected run construction summary

The final selected runs were constructed from a validated pre-final backbone plus small explicit patches.

Definitions:

- **Initial reference**: the first non-empty high-precision submission used as an anchor in the iterative experiments.
- **Backbone**: the best validated submission available at a given stage before adding a new patch.
- **Patch**: a small group of triples manually inspected and added to, or removed from, the backbone.

Final construction:

- The immediate pre-final backbone contained 929 triples.
- Run A adds three triples: two `Transmits` edges in document 102433 and one `Located_in` edge in document 100506, resulting in 932 triples.
- Run B adds Run A plus one additional `Located_in` hedge edge in document 100506, resulting in 933 triples.

The explicit patch specification is available in `patches/final_selected_run_patches.json`. The helper script `src/apply_final_patches.py` demonstrates how to apply such patches to a backbone CSV.
