# [🏁FASTER IMPLEMENTATION] 🦙TransLLaMa: LLM-based simultaneous translation

**Paper**: https://arxiv.org/pdf/2402.04636.pdf

![asdf](assets/output.gif)

# What is it?

🔥 This is a faster implmentation of TransLLaMa. If you have four A6000s or A100s on your machine, you can load the merged model (base Llama-2 with the LoRAs merged in) and run it much faster than the unmerged version (on the [main branch](https://github.com/RomanKoshkin/transllama/tree/main)). At the moment, you can use TransLLaMA-70B for `en-de`. For English-Russian (`en-ru`), only the slow unmerged version is available (feel free to open an inssue if you need the fast `en-ru` version).




# Quickstart

Follow all the steps listed in `README.md` on the main branch as necessary: install dependencies, specify everything needed in `.env`, prepare the datasets.

# Evaluation

## S2TT 

### (`en-de`) on `TED-TST-2023` (🦙70B)

```bash
cd evaluation
sh s_xp_ted2023_new_m.sh
```

### (`en-ru`) on `TED-TST-2023` (🦙70B)

Open an issue if you need it. Also, you can merge the model
