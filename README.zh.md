# SynthID Text

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

本仓库提供了发表在《自然》（_Nature_）期刊上的[研究论文][nature-paper]中 SynthID Text 大模型文本水印嵌入与检测能力的参考实现。本实现仅用于学术研究与复现，不适用于生产环境。核心代码库已[发布至 PyPI][synthid-pypi]，可在 [Python Notebook 示例][synthid-colab]中便捷安装运行，该示例完整演示了如何将这些工具应用于 [Gemma][gemma] 和 [GPT-2][gpt2] 模型。

## 安装与使用（Installation and usage）

配套的 [Colab Notebook][synthid-colab] 是一个自包含的端到端参考实现，涵盖：

1.  通过 [Mix-in 混入类][synthid-mixin] 扩展 [Hugging Face Transformers][transformers] 的 [`Gemma3ForCausalLM`][transformers-gemma3] 和 [`GPT2LMHeadModel`][transformers-gpt2] 类，支持为在 [PyTorch][pytorch] 中运行的模型生成的文本内容嵌入不可见水印；以及
2.  检测文本中嵌入的水印。既可使用无需训练的[频率学派检测器（Frequentist detector）][synthid-detector-frequentist]，也可使用功能更强大但需要提前[训练][synthid-detector-trainer]的[贝叶斯检测器（Bayesian detector）][synthid-detector-bayesian]。频率学派检测器针对“文本未采用给定密钥添加水印”的原假设返回精确的二项分布 p 值，因此无需针对不同文本长度手动微调判定阈值。

该 Notebook 旨在支持 Gemma 3 270M IT 或 GPT-2 模型端到端运行，在以下运行时硬件配置上表现最佳：

*   Gemma 3 270M IT：任意主流 GPU 均可（如 NVIDIA T4）。
*   GPT-2：任意运行时均可支持（高内存 CPU 或任意 GPU 速度更快）。

文本生成与 g 值计算均在 PyTorch 中执行，并在有可用 GPU 时自动调用硬件加速。贝叶斯检测器的训练与评分基于 JAX 实现，默认安装为 CPU 版本；由于检测器模型非常轻量，CPU 算力已完全足够。若希望在 NVIDIA GPU 上训练贝叶斯检测器，可安装 `cuda` 额外特性包，例如：`pip install synthid-text[notebook,cuda]`。

> **注意**：本实现仅供技术参考与研究复现目的使用。由于不同实现版本中 Gemma 模型结构存在微小差异，与原论文中报告的数值相比，本仓库测得的可检测性（Detectability）与困惑度（Perplexity）结果可能会有细微浮动。本文引入的子类并非针对生产系统设计。

> **注意**：本实现采用 20 轮 Threefry 2x32 基于计数器的伪随机数生成器（PRNG）进行生成时上下文哈希以及直接 Bernoulli(0.5) g 值位提取。Threefry 是一种经过充分研究、面向统计学质量设计的 PRNG，但它并非经过安全审计的密码学原语，且密钥长度限制为 63 位。本项目不提供任何密码学安全担保。

### 本地 Notebook 运行

若从源码安装，该 Notebook 也可在本地环境中运行。对于本地使用，强烈建议使用 Python 虚拟环境。

```shell
# 创建并激活虚拟环境
python3 -m venv ~/.venvs/synthid
source ~/.venvs/synthid/bin/activate

# 下载并安装 SynthID Text 及 Jupyter 运行环境
git clone https://github.com/google-deepmind/synthid-text.git
cd synthid-text
pip install '.[notebook-local]'

# 启动 Jupyter 服务器
python -m notebook
```

内核启动后，打开对应的 `.ipynb` Notebook 文件即可交互式执行。

### 运行测试套件

源码安装还包含了一组小型测试套件，用于验证库的各项功能是否符合预期。

```shell
# 创建并激活虚拟环境
python3 -m venv ~/.venvs/synthid
source ~/.venvs/synthid/bin/activate

# 从源码安装 SynthID Text 及测试依赖项
git clone https://github.com/google-deepmind/synthid-text.git
cd synthid-text
pip install '.[test]'

# 运行测试套件
pytest .
```

## 工作原理（How it works）

有关锦标赛水印机制（Tournament Watermarking）、基于双密钥水印的投机推测采样（Speculative Sampling）、检测算法原理、推荐测试用例及伪代码的详细说明，请参阅 [docs/tournament.md](./docs/tournament.md)。

### 定义水印配置

SynthID Text 根据给定的配置生成唯一水印，其中最核心的参数为秘密密钥 `key: int`（取值范围 `[0, 2**63)`）以及锦标赛层数 `depth: int`（锦标赛层次数，默认为 30）。

配置的结构由以下 `TypedDict` 子类定义，但在实践中，本库提供的 [Mix-in 混入类][synthid-mixin] 默认采用静态配置。

```python
from typing import TypedDict

import torch


class WatermarkingConfig(TypedDict):
    ngram_len: int
    key: int
    depth: int
    context_history_size: int
    device: torch.device
    skip_first_ngram_calls: bool
```

### 嵌入水印

水印通过包装 Transformers 库的 [`GemmaForCausalLM`][transformers-gemma]、[`Gemma3ForCausalLM`][transformers-gemma3]（纯文本 Gemma 3 模型，例如 Gemma 3 270M IT）以及 [`GPT2LMHeadModel`][transformers-gpt2] 类的 [Mix-in 混入类][synthid-mixin] 实现，生成与原生 Transformers 具有完全相同 API 的三个子类。
请注意：本库附带的 Mix-in 类使用静态水印配置，不适合直接在生产系统中使用。

该 Mix-in 要求使用 Transformers v5（`transformers>=5.16`）。水印是在 `generate()` 构建的所有 Logits 处理器和采样扭曲器（温度采样、top-k、top-p、重复惩罚等）之后应用的，因此它作用的正是模型原本将要采样的概率分布。
添加水印的文本生成要求配置 `do_sample=True` 且 `top_k >= 2`；其中 top-k 限制了每个时间步中参与水印评选的候选 Token 数量。默认配置中 `skip_first_ngram_calls=True`，因此前 `ngram_len - 1` 个生成的 Token 不会被添加水印；水印检测也不会对这些初始 Token 打分。

当指定 `return_dict_in_generate=True, output_scores=True` 时，返回的 `scores` 是水印嵌入处理后的分数，反映的是水印调整后的分布而非模型的原始分布（Mix-in 会对此发出警告）。如果需要模型的原始未加工 Logits（例如用于计算困惑度 Perplexity），请使用 `output_logits=True`。

```python
from synthid_text import synthid_mixin
import transformers
import torch


DEVICE = (
    torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
INPUTS = [
    "I enjoy walking with my cute dog",
    "I am from New York",
    "The test was not so very hard after all",
    "I don't think they can score twice in so short a time",
]
MODEL_NAME = 'google/gemma-3-270m-it'
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.99

# 从 Transformers 初始化标准分词器。Decoder-only 架构模型在批处理生成时需要使用左侧填充（Left Padding）。
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'left'
# 初始化启用了 SynthID Text 的模型。
model = synthid_mixin.SynthIDGemma3ForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    dtype=torch.bfloat16,
)
# 以常规方式准备输入数据。
inputs = tokenizer(
    INPUTS,
    return_tensors='pt',
    padding=True,
).to(DEVICE)
# 生成带有水印的文本。
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_length=1024,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
)
```

### 水印检测

水印检测可以通过多种评分函数实现（详见论文）。本仓库包含了论文中描述的**频率学派检测器（Frequentist Detector）**与**贝叶斯检测器（Bayesian Detector）**的代码。配套的 Colab 提供了这些检测器的使用示例。

以下两种检测器均接收移除了 Prompt 前缀的模型生成 Token ID，并在内部自动计算 g 值和有效掩码（Mask）。检测时所使用的 Logits 处理器必须与生成时配置完全一致的水印参数。

```python
from synthid_text import detector_frequentist
from synthid_text import logits_processing

logits_processor = logits_processing.SynthIDLogitsProcessor(
    **synthid_mixin.DEFAULT_WATERMARKING_CONFIG
)

# 仅从模型预测结果中截取新生成的文本部分（去除 Prompt）。
outputs = outputs[:, inputs['input_ids'].shape[1]:]
```

频率学派检测器无需任何训练。其 `p_value()` 函数针对每个样本返回“该文本未采用指定密钥添加水印”原假设的 p 值。较小的 p 值表示存在显著的水印证据。

```python
frequentist_detector = detector_frequentist.FrequentistDetector(
    logits_processor, eos_token_id=model.generation_config.eos_token_id
)
p_values = frequentist_detector.p_value(outputs)
```

贝叶斯检测器在投入使用前必须在带水印和无水印的数据上进行训练。贝叶斯检测器必须针对每个独立的水印密钥分别训练，且训练数据应独立于待测样本，同时能有效代表生产环境中系统预期生成文本的特征与风格质量。

```python
from synthid_text import detector_bayesian

# 形状为 [batch_size, length] 的 Token ID 数组列表（已去除 Prompt）：
# 分别包含带水印生成的响应，以及使用相同分词器分词的无水印文本。
tokenized_wm_outputs = ...
tokenized_uwm_outputs = ...

detector, selection_loss = detector_bayesian.BayesianDetector.train_best_detector(
    tokenized_wm_outputs=tokenized_wm_outputs,
    tokenized_uwm_outputs=tokenized_uwm_outputs,
    logits_processor=logits_processor,
    eos_token_id=model.generation_config.eos_token_id,
    torch_device=DEVICE,
)
```

贝叶斯检测器训练完成后，使用 `detector.score()` 函数即可为每个样本生成一个评分，指示文本是否由给定的水印配置生成。评分取值范围在 0 到 1 之间，数值越接近 1，表示文本由给定水印生成的可能性越高。你可以根据业务准确率与误报率要求灵活调整判决阈值。

```python
bayesian_scores = detector.score(outputs)
```

此外，也可以直接提取各位置上的 g 值以及参与评分的位置掩码：

```python
g_values, mask = logits_processor.compute_g_values_and_mask(
    outputs, eos_token_id=model.generation_config.eos_token_id
)
```

## 人工评估数据集（Human Data）

我们公开发布了人工评估数据集，用于对比 Gemma 7B 模型生成的带水印文本与无水印文本。数据集位于 `data/human_eval.jsonl`。如需获取生成这些响应所使用的原始提示词（Prompts），请使用以下代码加载。

```python
import json
import tensorflow_datasets as tfds

ds = tfds.load('huggingface:eli5/LFQA_reddit', split='test_eli5')
id_to_prompt = {}
for x in ds.as_numpy_iterator():
  id_to_prompt[x['q_id'].decode()] = x['title'].decode()

full_data = []
with open('./data/human_eval.jsonl') as f:
  for json_str in f:
    x = json.loads(json_str)
    x['question'] = id_to_prompt[x['q_id']]
    full_data.append(x)
```

## 双密钥投机解码（Dual-Key Speculative Decoding）

`speculative_engine` 模块提供了基于双密钥锦标赛水印的投机推测采样（Speculative Sampling）独立实现。它通过将目标模型（Target Model）与一个共享分词器的轻量草稿模型（Draft Model）协同工作来显著加速生成：对被接受的草稿 Token 使用 `key_draft` 嵌入水印，而对替换/奖励 Token 使用 `key_target` 嵌入水印。在所有水印密钥上平均后，每个生成的 Token 均严格遵循目标模型的 top-k 分布，与标准的自回归 top-k 采样统计一致。有关直观展示 Token 来源、接受率与检测器 $p$ 值的交互式演练，请参阅 [Colab 中的投机解码 Notebook][synthid-speculative-colab]。

```python
from synthid_text import detector_frequentist
from synthid_text import speculative_engine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
target_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

# 水印逻辑在 CPU 上执行（watermark_device 默认为 CPU），而模型在 `device` 上运行。
config = speculative_engine.DualKeyWatermarkingConfig(
    ngram_len=5,
    key_draft=42,
    key_target=84,
    depth=15,
    temperature=0.7,
    top_k=40,
    draft_len=4,
    target_device=device,
    draft_device=device,
)

engine = speculative_engine.DualKeySpeculativeEngine(
    target_model=target_model,
    draft_model=draft_model,
    config=config,
)

# 使用双密钥投机解码生成带水印的文本。引擎单次生成一个序列（batch size 为 1）。
prompt_ids = tokenizer(
    "Write a short story about a lighthouse.", return_tensors="pt"
).input_ids
output = engine.generate(
    prompt_ids,
    max_new_tokens=100,
    eos_token_id=target_model.generation_config.eos_token_id,
)
print(tokenizer.decode(output.sequences[0]))

# 聚合草稿密钥与目标密钥信号的统计检测。仅对新生成的 Token 评分；
# output.sequences 中同样包含了前缀 Prompt。
generated_ids = output.sequences[:, prompt_ids.shape[1]:]
detector = detector_frequentist.DualKeyFrequentistDetector.from_config(
    config, eos_token_id=target_model.generation_config.eos_token_id
)
p_value = detector.p_value(generated_ids)[0]
print(f"p-value: {p_value:.4e}")
```

## 引用（Citing this work）

如果您在研究中使用了本技术或代码，请引用我们在《自然》期刊发表的论文：

```bibtex
@article{Dathathri2024,
    author={Dathathri, Sumanth and See, Abigail and Ghaisas, Sumedh and Huang, Po-Sen and McAdam, Rob and Welbl, Johannes and Bachani, Vandana and Kaskasoli, Alex and Stanforth, Robert and Matejovicova, Tatiana and Hayes, Jamie and Vyas, Nidhi and Merey, Majd Al and Brown-Cohen, Jonah and Bunel, Rudy and Balle, Borja and Cemgil, Taylan and Ahmed, Zahra and Stacpoole, Kitty and Shumailov, Ilia and Baetu, Ciprian and Gowal, Sven and Hassabis, Demis and Kohli, Pushmeet},
    title={Scalable watermarking for identifying large language model outputs},
    journal={Nature},
    year={2024},
    month={Oct},
    day={01},
    volume={634},
    number={8035},
    pages={818-823},
    issn={1476-4687},
    doi={10.1038/s41586-024-08025-4},
    url={https://doi.org/10.1038/s41586-024-08025-4}
}
```

## 许可证与免责声明（License and disclaimer）

版权所有 2024 DeepMind Technologies Limited

所有软件均基于 Apache License 2.0（Apache 2.0）许可证授权；您必须在遵守 Apache 2.0 许可证的前提下使用本软件。您可以在以下地址获取许可证副本：
https://www.apache.org/licenses/LICENSE-2.0

所有其他材料均采用知识共享署名 4.0 国际许可协议（CC-BY）授权。您可以在以下地址获取 CC-BY 许可证副本：
https://creativecommons.org/licenses/by/4.0/legalcode

除非适用法律明确要求或书面同意，根据 Apache 2.0 或 CC-BY 许可证分发的软件和材料均按“原样”分发，不附带任何明示或暗示的担保或条件。有关许可证下的具体权限与限制，请参阅对应许可证文本。

本项目并非 Google 官方正式产品。

[flax]: https://github.com/google/flax
[gemma]: https://ai.google.dev/gemma/docs/model_card
[gpt2]: https://huggingface.co/openai-community/gpt2
[jax]: https://github.com/jax-ml/jax
[pytorch]: https://pytorch.org/
[nature-paper]: https://doi.org/10.1038/s41586-024-08025-4
[synthid-colab]: https://colab.research.google.com/github/google-deepmind/synthid-text/blob/main/notebooks/synthid_text_huggingface_integration.ipynb
[synthid-speculative-colab]: https://colab.research.google.com/github/google-deepmind/synthid-text/blob/main/notebooks/synthid_text_speculative_decoding.ipynb
[synthid-pypi]: https://pypi.org/project/synthid-text/
[synthid-detector-bayesian]: ./src/synthid_text/detector_bayesian.py
[synthid-detector-frequentist]: ./src/synthid_text/detector_frequentist.py
[synthid-detector-trainer]: ./src/synthid_text/detector_bayesian.py
[synthid-mixin]: ./src/synthid_text/synthid_mixin.py
[transformers]: https://github.com/huggingface/transformers
[transformers-gemma]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
[transformers-gemma3]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
[transformers-gpt2]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月24日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
