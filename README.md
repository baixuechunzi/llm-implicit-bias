## Measuring Implicit Bias in Explicitly Unbiased Large Language Models

[Xuechunzi Bai](https://www.xuechunzibai.com/)<sup>$\S$</sup>, 
[Angelina Wang](https://angelina-wang.github.io/)<sup>$\dagger$</sup>, 
[Ilia Sucholutsky](https://ilia10000.github.io/)<sup>$\dagger$</sup>, 
[Thomas L. Griffiths](https://cocosci.princeton.edu/tom/index.php)<sup>$\S$</sup>,<sup>$\dagger$</sup>

<sup>$\S$</sup>Department of Psychology,
<sup>$\dagger$</sup>Department of Computer Science,
Princeton University
<br>

---
### Overview
<br>

> Large language models (LLMs) can pass explicit bias tests but still harbor implicit biases, similar to humans who endorse egalitarian beliefs yet exhibit subtle biases.
> Informed by psychology, we introduce two measures that effectively expose implicit biases and subtle discrimination in proprietary LLMs that do not show explicit bias on standard benchmarks ([see below, A Quick Glance](#section1)).
> 
> We found pervasive human-like stereotype biases in 8 LLMs (GPT-4, GPT-3.5-Turbo, Claude2-Opus, Claude3-Sonnet, LLaMA2-70B, LLaMA2-13B, LLaMA2-7B, Alpaca-7B), plus GPT-4o, across 4 social categories (race, gender, religion, health) and 21 stereotypes (e.g., race and negativity, race and weapon, gender and power, gender and science, age and negativity, etc).

> Read more: **[[arXiv](https://arxiv.org/abs/2402.04105)]**
<br>

---
<a name="section1"></a>
### A Quick Glance
<br>

> 💡 Here is an example of LLM Implicit Bias (middle panel) and Decision Bias (right panel) in explicitly unbiased GPT-4 (left panel).
<br>

> ![Figure1](demo.png)
<br>

---

> A note: This repo provides prompts and analysis code for reproducibility. For easier interpretation, we have reorganized the code into smaller chunks. We also release raw experimental data that can be reanalyzed for further scientific questions.
> 
