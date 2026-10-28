# vectfit

Rational macromodels of multiport frequency responses in Rust.

It fits sampled responses (S, Y or Z parameters over frequency) with a common set of poles, using fast relaxed vector fitting (Gustavsen and Semlyen; relaxed pole relocation after Gustavsen; multiport after Deschrijver). The model order is chosen automatically by adding and skimming poles (Grivet-Talocia and Bandinu). The models can be checked and enforced for passivity, and exported as deterministic Verilog-A.

The crate depends only on `faer`, `num-complex` and `rayon`. The fit is bit-identical across thread counts.

It is a Rust port of the Python implementation [Vectorfitting](https://github.com/milanofthe/Vectorfitting).

## License

MIT
