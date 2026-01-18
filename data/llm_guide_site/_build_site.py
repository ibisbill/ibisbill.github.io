#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a multi-page interactive website from `Transformer_LLM_Study_Guide_CN.pdf`.

Outputs:
  data/llm_guide_site/index.html
  data/llm_guide_site/*.html (module pages + key knowledge pages)
  data/llm_guide_site/source_extracted.txt (for traceability)

Design goals:
  - static HTML (GitHub Pages friendly)
  - MathJax for formulas
  - highlight.js for code blocks
  - vanilla JS demos via assets/site.js
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ibisbill.github.io
VENDOR = REPO_ROOT / ".vendor"
PDF_PATH = REPO_ROOT / "data/transformers/Transformer_LLM_Study_Guide_CN.pdf"
OUT_DIR = REPO_ROOT / "data/llm_guide_site"


def _ensure_pdfminer():
  sys.path.insert(0, str(VENDOR))
  from pdfminer.high_level import extract_text  # noqa: F401


@dataclass
class Section:
  id: str            # e.g. "2.1"
  title: str         # e.g. "Scaled Dot-Product Attention"
  level: int         # 0=module, 1=subsection
  lines: List[str]


def extract_pdf_text() -> str:
  _ensure_pdfminer()
  from pdfminer.high_level import extract_text
  return extract_text(str(PDF_PATH))


def normalize_lines(text: str) -> List[str]:
  lines = [ln.rstrip() for ln in text.splitlines()]
  # remove page markers like "第 12 页"
  out: List[str] = []
  for ln in lines:
    if re.match(r"^第\s*\d+\s*页$", ln.strip()):
      continue
    # skip pure form-feed remnants (pdfminer often inserts \f as empty lines)
    if ln.strip() == "\f":
      continue
    out.append(ln)
  return out


def find_content_start(lines: List[str]) -> int:
  # The real content begins after the TOC pages. We locate the first module header
  # that is followed soon by "学习目标：" to avoid the TOC occurrence.
  for i in range(len(lines)):
    if lines[i].strip() == "0 预备基础":
      window = "\n".join(lines[i:i+30])
      if "学习目标" in window:
        return i
  # fallback: first occurrence of "学习目标："
  for i, ln in enumerate(lines):
    if "学习目标" in ln:
      return max(0, i - 2)
  return 0


def parse_sections(lines: List[str]) -> List[Section]:
  start = find_content_start(lines)
  lines = lines[start:]

  # headings:
  # module: "2 Transformer 核心结构"
  # sub: "2.1 Scaled Dot-Product Attention"
  re_module = re.compile(r"^(\d+)\s+(.+?)\s*$")
  re_sub = re.compile(r"^(\d+\.\d+)\s+(.+?)\s*$")

  sections: List[Section] = []
  cur: Optional[Section] = None

  def flush():
    nonlocal cur
    if cur is not None:
      # trim leading/trailing empties
      while cur.lines and cur.lines[0].strip() == "":
        cur.lines.pop(0)
      while cur.lines and cur.lines[-1].strip() == "":
        cur.lines.pop()
      sections.append(cur)
      cur = None

  for ln in lines:
    s = ln.strip()
    if not s:
      if cur is not None:
        cur.lines.append("")
      continue

    m_sub = re_sub.match(s)
    m_mod = re_module.match(s)
    if m_sub:
      flush()
      cur = Section(id=m_sub.group(1), title=m_sub.group(2), level=1, lines=[])
      continue
    if m_mod:
      # avoid capturing numbered questions like "Q1：..."
      if s.startswith("Q") and "：" in s:
        if cur is not None:
          cur.lines.append(s)
        continue
      flush()
      cur = Section(id=m_mod.group(1), title=m_mod.group(2), level=0, lines=[])
      continue

    if cur is None:
      # ignore prelude noise
      continue
    cur.lines.append(ln)

  flush()
  return sections


def section_to_html(lines: List[str]) -> str:
  """
  Convert the PDF-ish plain text into lightweight HTML.
  - bullet lists: "- xxx"
  - Q/A blocks: "Q1：..." and "A1：..."
  """
  html: List[str] = []
  in_ul = False

  def close_ul():
    nonlocal in_ul
    if in_ul:
      html.append("</ul>")
      in_ul = False

  qa_buf: List[str] = []
  def flush_qa():
    nonlocal qa_buf
    if not qa_buf:
      return
    q = ""
    a = ""
    for ln in qa_buf:
      if ln.startswith("Q"):
        q = ln
      elif ln.startswith("A"):
        a = ln
    html.append('<div class="qa">')
    if q:
      html.append(f'<div class="q">{escape(q)}</div>')
    if a:
      html.append(f'<div class="a">{escape(a)}</div>')
    html.append("</div>")
    qa_buf = []

  for raw in lines:
    ln = raw.strip("\n")
    s = ln.strip()
    if s == "":
      close_ul()
      flush_qa()
      continue

    if re.match(r"^Q\d+：", s) or re.match(r"^A\d+：", s):
      close_ul()
      qa_buf.append(s)
      # if we collected both Q and A, flush
      if any(x.startswith("Q") for x in qa_buf) and any(x.startswith("A") for x in qa_buf):
        flush_qa()
      continue

    if s.startswith("- "):
      flush_qa()
      if not in_ul:
        html.append("<ul>")
        in_ul = True
      html.append(f"<li>{escape(s[2:])}</li>")
      continue

    close_ul()
    flush_qa()
    html.append(f"<p>{escape(s)}</p>")

  close_ul()
  flush_qa()
  return "\n".join(html)


def page_template(*, title: str, subtitle: str, body_html: str, nav_active: str = "", pager: Tuple[Optional[Tuple[str,str]], Optional[Tuple[str,str]]] = (None, None)) -> str:
  prev_link, next_link = pager
  def pager_a(link):
    if not link:
      return '<span style="flex:1"></span>'
    href, text = link
    return f'<a href="{escape(href)}"><strong>{escape(text)}</strong></a>'

  return f"""<!doctype html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="./assets/site.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css" />
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$','$'], ['\\\\(','\\\\)']],
        displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
      }},
      options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/highlight.min.js"></script>
  <script defer src="./assets/site.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', () => {{
      if (window.hljs) window.hljs.highlightAll();
    }});
  </script>
</head>
<body>
  <div class="topbar">
    <div class="container topbar-inner">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <div>Transformer × LLM 学习站</div>
          <div class="brand-sub">{escape(subtitle)}</div>
        </div>
      </div>
      <div class="navlinks">
        <a href="./index.html" class="{ 'active' if nav_active=='index' else '' }">目录</a>
        <a href="./module-02-transformer.html" class="{ 'active' if nav_active=='m2' else '' }">Transformer</a>
        <a href="./module-07-alignment.html" class="{ 'active' if nav_active=='m7' else '' }">对齐</a>
        <a href="./module-08-inference.html" class="{ 'active' if nav_active=='m8' else '' }">推理</a>
      </div>
    </div>
  </div>

  <div class="container">
    {body_html}
    <div class="pager">
      {pager_a(prev_link)}
      {pager_a(next_link)}
    </div>
    <div class="footer">
      <div>来源：`Transformer_LLM_Study_Guide_CN.pdf`（生成日期 2026-01-18）</div>
      <div><a href="../transformers/Transformer_LLM_Study_Guide_CN.pdf">打开原 PDF</a></div>
    </div>
  </div>
</body>
</html>
"""


def write(path: Path, content: str):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding="utf-8")


def slug_for_section(sec_id: str, title: str, is_module: bool) -> str:
  if is_module:
    return f"module-{int(sec_id):02d}-{_slugify(title)}.html"
  return f"kp-{sec_id.replace('.','-')}-{_slugify(title)}.html"


def _slugify(s: str) -> str:
  s = s.strip().lower()
  s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s)
  s = re.sub(r"-{2,}", "-", s).strip("-")
  if not s:
    return "page"
  return s[:60]


def extra_blocks() -> Dict[str, str]:
  """Hand-authored high-signal additions: formulas, code walkthroughs, demos, references."""
  return {
    "2.1": """
<h3>核心公式：Scaled Dot-Product Attention</h3>
<p>给定 $Q \\in \\mathbb{R}^{S\\times d_k}$、$K \\in \\mathbb{R}^{S\\times d_k}$、$V \\in \\mathbb{R}^{S\\times d_v}$：</p>
<p>$$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$</p>
<p class="muted">关键点：$\\sqrt{d_k}$ 缩放用于控制 dot-product 的方差，避免 softmax 在训练初期过度饱和。</p>

<div class="demo" data-demo="attention">
  <h3>交互 Demo：手算 Attention（含中间矩阵）</h3>
  <div class="row">
    <div class="col">
      <label>Q（JSON 2D 数组，形状 S×d_k）</label>
      <textarea data-role="Q"></textarea>
    </div>
    <div class="col">
      <label>K（JSON 2D 数组，形状 S×d_k）</label>
      <textarea data-role="K"></textarea>
    </div>
    <div class="col">
      <label>V（JSON 2D 数组，形状 S×d_v）</label>
      <textarea data-role="V"></textarea>
    </div>
  </div>
  <div class="row">
    <button class="btn primary" data-action="run">运行：计算 Scores/Probs/Out</button>
    <button class="btn" data-action="reset">重置示例</button>
  </div>
  <div class="out" data-role="OUT"></div>
</div>

<h3>代码解读（PyTorch 形状视角）</h3>
<div class="codeblock">
  <div class="codehdr"><span>伪代码（接近 PyTorch 实现）</span><span class="kbd">B=batch, S=seq, H=heads, D=head_dim</span></div>
  <pre><code class="language-python"># q,k,v: (B, H, S, D)
scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)   # (B, H, S, S)
probs  = scores.softmax(dim=-1)                     # (B, H, S, S)
out    = probs @ v                                  # (B, H, S, D)
</code></pre>
</div>

<h3>原论文与典型代码库</h3>
<ul>
  <li>论文：Vaswani et al., 2017, <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></li>
  <li>代码（参考实现）：<a href="https://github.com/huggingface/transformers">huggingface/transformers</a>（多模型 attention 实现）</li>
  <li>代码（极简教学）：<a href="https://github.com/karpathy/nanoGPT">karpathy/nanoGPT</a></li>
</ul>
""",
    "1.3": """
<h3>正弦位置编码（Sinusoidal PE）</h3>
<p>原 Transformer 使用固定位置编码：</p>
<p>$$PE(pos,2i)=\\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right),\\quad PE(pos,2i+1)=\\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$</p>
<p class="muted">直觉：不同频率的正弦/余弦提供“可外推”的相对位置信息；线性组合可近似位移。</p>

<div class="demo" data-demo="posenc">
  <h3>交互 Demo：生成并可视化某个位置的 PE</h3>
  <div class="row">
    <div class="col">
      <label>pos</label>
      <input data-role="pos" />
    </div>
    <div class="col">
      <label>d_model</label>
      <input data-role="dim" />
    </div>
  </div>
  <div class="row">
    <button class="btn primary" data-action="run">生成</button>
  </div>
  <canvas width="920" height="220" style="width:100%;margin-top:10px;border-radius:14px;border:1px solid rgba(255,255,255,.10);background:rgba(0,0,0,.18)"></canvas>
  <div class="out" data-role="OUT"></div>
</div>

<h3>扩展阅读：RoPE/ALiBi</h3>
<ul>
  <li>RoPE：Su et al., 2021, <a href="https://arxiv.org/abs/2104.09864">RoFormer</a></li>
  <li>ALiBi：Press et al., 2021, <a href="https://arxiv.org/abs/2108.12409">Train Short, Test Long</a></li>
</ul>
""",
    "1.2": """
<h3>Tokenization：BPE / SentencePiece / Byte-level</h3>
<p class="muted">Tokenizer 决定了“一个 token 是什么”，进而影响长度、成本、以及 PPL 的可比性。</p>

<h3>代码解读要点（读实现时看这些）</h3>
<ul>
  <li><b>归一化</b>：NFKC、大小写、空白、特殊符号处理（对中英文混排尤其关键）</li>
  <li><b>预分词</b>：按空白/标点切分或 byte-level 直接从字节开始</li>
  <li><b>合并规则</b>：BPE merge table / unigram LM 的概率模型</li>
  <li><b>特殊 token</b>：BOS/EOS/PAD/UNK，以及对话模板中的 role 标记</li>
</ul>

<h3>原论文与代码库</h3>
<ul>
  <li>BPE：Sennrich et al., 2016 <a href="https://arxiv.org/abs/1508.07909">Neural Machine Translation of Rare Words with Subword Units</a></li>
  <li>SentencePiece：Kudo & Richardson, 2018 <a href="https://arxiv.org/abs/1808.06226">SentencePiece</a></li>
  <li>字节级 BPE（GPT-2）：Radford et al., 2019（见 GPT-2 tokenizer 设计）</li>
  <li>实现：<a href="https://github.com/huggingface/tokenizers">huggingface/tokenizers</a>、<a href="https://github.com/google/sentencepiece">google/sentencepiece</a></li>
</ul>
""",
    "2.2": """
<h3>Multi-Head Attention（MHA）</h3>
<p>核心思想：用多个 head 在不同子空间并行做 attention，然后拼接回去。</p>
<p>$$\\mathrm{head}_i=\\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$</p>
<p>$$\\mathrm{MHA}(Q,K,V)=\\mathrm{Concat}(\\mathrm{head}_1,\\dots,\\mathrm{head}_h)W^O$$</p>
<p class="muted">实现里常见的张量变换：先线性投影到 (B,S,H*D)，再 reshape/transpose 到 (B,H,S,D)。</p>

<h3>原论文与代码库</h3>
<ul>
  <li>论文：Vaswani et al., 2017 <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></li>
  <li>代码：<a href="https://github.com/huggingface/transformers">huggingface/transformers</a>（如 `modeling_llama.py`/`modeling_gpt2.py` 的 attention 投影与 reshape）</li>
</ul>
""",
    "7.3": """
<h3>DPO：直接偏好优化（Direct Preference Optimization）</h3>
<p>DPO 用“偏好对”直接优化策略模型，避免显式训练 reward model + PPO 的复杂度。一个常见写法是：</p>
<p>$$\\mathcal{L}_{\\mathrm{DPO}}(\\theta)= -\\mathbb{E}_{(x,y^+,y^-)}\\left[\\log \\sigma\\left(\\beta\\left(\\log \\pi_\\theta(y^+|x)-\\log \\pi_\\theta(y^-|x) - \\log \\pi_{\\mathrm{ref}}(y^+|x)+\\log \\pi_{\\mathrm{ref}}(y^-|x)\\right)\\right)\\right]$$</p>
<p class="muted">$\\pi_{ref}$ 通常是 SFT 模型；括号里是“相对偏好优势”，$\\beta$ 控制对齐强度。</p>

<h3>工程实现要点（读代码时关注）</h3>
<ul>
  <li>如何计算 $\\log \\pi(y|x)$：通常是 token-level logprob 求和（只对回答区间）</li>
  <li>mask/packing：对话模板导致不同样本长度差异，必须正确 mask padding</li>
  <li>稳定性：对 logit 做 clamp、对 $\\beta$ 做调参，避免训练早期过强偏好</li>
</ul>

<h3>原论文与典型代码库</h3>
<ul>
  <li>论文：Rafailov et al., 2023, <a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization</a></li>
  <li>代码：<a href="https://github.com/huggingface/trl">huggingface/trl</a>（包含 DPOTrainer 等实现）</li>
  <li>对齐背景：Ouyang et al., 2022, <a href="https://arxiv.org/abs/2203.02155">InstructGPT (RLHF)</a></li>
</ul>
""",
    "7.2": """
<h3>RLHF 三阶段：SFT → Reward Model → PPO</h3>
<ul>
  <li><b>SFT</b>：用高质量指令数据把模型拉到“能听话、能按格式回答”的区域。</li>
  <li><b>RM</b>：用偏好对训练奖励模型（更偏好 y+、更不偏好 y-）。</li>
  <li><b>PPO</b>：在 KL 约束下优化策略模型，使其在 RM 上得分更高且不偏离参考模型太远。</li>
</ul>

<h3>PPO 的一个关键约束（直觉）</h3>
<p class="muted">实际工程里常见项：$\\beta\\,\\mathrm{KL}(\\pi_\\theta\\,||\\,\\pi_{ref})$，避免策略“跑飞”导致语言质量/稳定性下降。</p>

<h3>原论文与代码库</h3>
<ul>
  <li>InstructGPT（RLHF）：Ouyang et al., 2022 <a href="https://arxiv.org/abs/2203.02155">Training language models to follow instructions</a></li>
  <li>PPO：Schulman et al., 2017 <a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a></li>
  <li>实现：<a href="https://github.com/huggingface/trl">huggingface/trl</a></li>
</ul>
""",
    "8.3": """
<h3>解码策略小抄：Greedy / Beam / Sampling</h3>
<ul>
  <li><b>Greedy</b>：每步选 argmax，最快但多样性差，易陷入重复</li>
  <li><b>Beam Search</b>：保留 top-B 条候选序列，适合机器翻译等“确定性”任务；对开放式生成可能变“更无聊”</li>
  <li><b>Sampling</b>：按概率采样（温度、top-k、top-p），更有创造性但方差更大</li>
</ul>

<h3>温度与 top-k / top-p 的数学表达</h3>
<p>给定 logits $z$，温度 $T$：$p_i \\propto \\exp(z_i/T)$。$T\\uparrow$ 更“随机”，$T\\downarrow$ 更“确定”。</p>
<p class="muted">top-k / top-p 会把低概率 token 置零再重归一化（改变分布尾部）。</p>

<h3>参考实现代码库</h3>
<ul>
  <li><a href="https://github.com/huggingface/transformers">huggingface/transformers</a>：`generate()` 覆盖 greedy/beam/sampling/contrastive 等</li>
  <li><a href="https://github.com/vllm-project/vllm">vllm-project/vllm</a>：高吞吐推理与采样实现</li>
</ul>
""",
    "3.1": """
<h3>Causal LM（自回归语言建模）目标</h3>
<p>给定 token 序列 $x_{1:T}$，最大似然训练等价于最小化负对数似然：</p>
<p>$$\\mathcal{L}_{\\mathrm{CLM}}(\\theta) = -\\sum_{t=1}^{T}\\log p_\\theta(x_t\\mid x_{&lt;t})$$</p>
<p class="muted">实现里通常是 token-level cross entropy：对每个位置的 logits 做 softmax，再取真实 token 的 $-\\log p$，最后对有效位置求均值。</p>

<h3>困惑度（Perplexity）与 bits-per-token</h3>
<p>若平均交叉熵为 $H$（单位 nat），则：</p>
<p>$$\\mathrm{PPL}=\\exp(H)$$</p>
<p>若用 bit 表示：$\\mathrm{bpt}=H/\\ln 2$。</p>
<p class="muted">注意：不同 tokenizer 的 PPL 不能直接横比（手册里也提醒了这一点）。</p>

<h3>原论文与典型代码库</h3>
<ul>
  <li>GPT 系列（自回归 LM）：Radford et al., 2018/2019；Brown et al., 2020 <a href="https://arxiv.org/abs/2005.14165">Language Models are Few-Shot Learners</a></li>
  <li>代码：<a href="https://github.com/huggingface/transformers">huggingface/transformers</a>（`CausalLMOutputWithCrossAttentions` 与 loss 计算）</li>
  <li>教学代码：<a href="https://github.com/karpathy/nanoGPT">karpathy/nanoGPT</a>（最清晰的 CE loss + next-token shift）</li>
</ul>
""",
    "8.2": """
<h3>KV Cache：为什么能加速 Decode？</h3>
<p>在 decode 阶段，每次只生成 1 个新 token。若不缓存，每步都要对整段历史重新计算 K/V；KV cache 把历史 token 的 K/V 保留下来，使得每步只需计算新 token 的 K/V 并与历史做注意力。</p>

<h3>显存估算（近似）</h3>
<p>对 decoder-only：每层缓存 K 与 V，形状近似为 $(B, H, S, D)$。若用 FP16/BF16（2 bytes）：</p>
<p>$$\\mathrm{KV\\_bytes}\\approx 2\\,(K+V)\\times B\\times L\\times H\\times S\\times D\\times bytes$$</p>
<p class="muted">这也是长上下文推理最“吃显存”的来源之一。</p>

<div class="demo" data-demo="kvcalc">
  <h3>交互 Demo：KV Cache 显存计算器</h3>
  <div class="row">
    <div class="col"><label>Batch B</label><input data-role="B" value="1" /></div>
    <div class="col"><label>Layers L</label><input data-role="L" value="32" /></div>
    <div class="col"><label>Heads H</label><input data-role="H" value="32" /></div>
    <div class="col"><label>Head dim D</label><input data-role="D" value="128" /></div>
    <div class="col"><label>Seq len S</label><input data-role="S" value="4096" /></div>
    <div class="col"><label>dtype bytes（FP16=2, FP8=1）</label><input data-role="bytes" value="2" /></div>
  </div>
  <div class="row">
    <button class="btn primary" data-action="run">计算</button>
  </div>
  <div class="out" data-role="OUT"></div>
</div>

<h3>原论文与代码库</h3>
<ul>
  <li>KV cache 的系统化实现可参考：<a href="https://github.com/vllm-project/vllm">vLLM</a>（PagedAttention 等）</li>
  <li>推理与服务工程：<a href="https://github.com/huggingface/text-generation-inference">huggingface/text-generation-inference</a></li>
</ul>
""",
    "2.6": """
<h3>结构演进速览：MHA → MQA/GQA、FlashAttention、MoE</h3>
<ul>
  <li><b>MQA/GQA</b>：减少 K/V 的 head 数（共享/分组），降低 KV cache 显存与带宽压力。</li>
  <li><b>FlashAttention</b>：通过分块与 IO-aware 的实现，把注意力计算做成更少显存读写、更快的 kernel。</li>
  <li><b>MoE</b>：用路由器选择少量专家前馈网络，提升参数规模而控制计算量。</li>
</ul>

<h3>原论文与典型代码库</h3>
<ul>
  <li>FlashAttention：Dao et al., 2022 <a href="https://arxiv.org/abs/2205.14135">FlashAttention</a>；代码：<a href="https://github.com/Dao-AILab/flash-attention">Dao-AILab/flash-attention</a></li>
  <li>MQA：Shazeer, 2019 <a href="https://arxiv.org/abs/1911.02150">Fast Transformer Decoding</a></li>
  <li>GQA：Ainslie et al., 2023 <a href="https://arxiv.org/abs/2305.13245">GQA</a></li>
  <li>MoE：Fedus et al., 2021 <a href="https://arxiv.org/abs/2101.03961">Switch Transformers</a></li>
  <li>工程实现：<a href="https://github.com/huggingface/transformers">huggingface/transformers</a>（GQA/RoPE/MoE 等多种变体）</li>
</ul>
""",
    "9.2": """
<h3>RoPE：旋转位置编码（Rotary Position Embedding）</h3>
<p>RoPE 的核心是把每个 head 的向量按维度成对分组，在复平面上做位置相关旋转，从而把“相对位置”融入 dot-product。</p>
<p class="muted">工程上你通常会在实现里看到：对 q/k 做 rotate-half、按频率表 cos/sin 做融合（LLaMA 等）。</p>

<h3>扩展/外推：为什么要做 RoPE scaling？</h3>
<p>训练时上下文长度有限，测试时拉长会导致频率分布与相位偏移失配；所以会出现各种 scaling / interpolation / yarn 类技巧来改善长上下文外推。</p>

<h3>原论文与代码库</h3>
<ul>
  <li>RoPE：Su et al., 2021 <a href="https://arxiv.org/abs/2104.09864">RoFormer</a></li>
  <li>ALiBi：Press et al., 2021 <a href="https://arxiv.org/abs/2108.12409">Train Short, Test Long</a></li>
  <li>参考实现：<a href="https://github.com/meta-llama/llama">meta-llama/llama</a>、<a href="https://github.com/huggingface/transformers">huggingface/transformers</a></li>
</ul>
""",
    "10.2": """
<h3>ReAct：Reason + Act 的 Agent 范式</h3>
<p>ReAct 让模型交替输出“思考（Reason）”与“行动（Act）”，通过工具/检索获得外部信息，再继续推理与执行。</p>
<p class="muted">工程关键：把工具调用结果纳入可观测 trace；对 prompt injection 做隔离；把权限收敛到最小。</p>

<h3>原论文与代码库</h3>
<ul>
  <li>ReAct：Yao et al., 2022 <a href="https://arxiv.org/abs/2210.03629">ReAct</a></li>
  <li>典型实现：<a href="https://github.com/langchain-ai/langchain">langchain-ai/langchain</a>、<a href="https://github.com/run-llama/llama_index">run-llama/llama_index</a></li>
</ul>
""",
    "11.1": """
<h3>量化：为什么能显著提速/省显存？</h3>
<ul>
  <li><b>权重量化</b>：把权重从 FP16/FP32 压到 INT8/INT4（或更低），减小模型体积与带宽。</li>
  <li><b>激活量化</b>：对推理中间激活做量化（更难），可进一步提速但更敏感。</li>
  <li><b>KV 量化</b>：长上下文推理下收益非常明显（KV cache 往往占大头）。</li>
</ul>

<h3>原论文与代码库</h3>
<ul>
  <li>GPTQ：Frantar et al., 2022 <a href="https://arxiv.org/abs/2210.17323">GPTQ</a></li>
  <li>AWQ：Lin et al., 2023 <a href="https://arxiv.org/abs/2306.00978">AWQ</a></li>
  <li>工程实现：<a href="https://github.com/ggerganov/llama.cpp">ggerganov/llama.cpp</a>、<a href="https://github.com/bitsandbytes-foundation/bitsandbytes">bitsandbytes</a></li>
</ul>
""",
    "13.3": """
<h3>机制可解释性：Activation patching / Causal tracing / SAE</h3>
<p class="muted">手册里强调：attention heatmap 不是因果解释；需要 patching/ablation 这样的干预来验证“因果贡献”。</p>

<h3>SAE（稀疏自编码器）是什么？</h3>
<p>SAE 试图把高维激活分解为一组稀疏可解释特征：$x \\approx W h$，其中 $h$ 稀疏（L1 或稀疏约束）。直觉上，每个 feature 更接近“单一语义”。</p>

<h3>参考论文与项目</h3>
<ul>
  <li>Dictionary Learning / SAE 路线：<a href="https://arxiv.org/abs/2309.10312">Towards Monosemanticity</a>（Anthropic, 2023）</li>
  <li>工具与实现（社区）：<a href="https://github.com/TransformerLensOrg/TransformerLens">TransformerLens</a></li>
</ul>
""",
  }


def build():
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  text = extract_pdf_text()
  write(OUT_DIR / "source_extracted.txt", text)

  lines = normalize_lines(text)
  secs = parse_sections(lines)

  # Split modules and subsections
  modules: Dict[str, Section] = {}
  subs_by_module: Dict[str, List[Section]] = {}
  for s in secs:
    if s.level == 0:
      modules[s.id] = s
      subs_by_module.setdefault(s.id, [])
    else:
      mod = s.id.split(".")[0]
      subs_by_module.setdefault(mod, []).append(s)

  extras = extra_blocks()

  # Build a page list order for prev/next
  module_ids = sorted(modules.keys(), key=lambda x: int(x))

  # Map id -> filename
  module_file: Dict[str, str] = {}
  sub_file: Dict[str, str] = {}
  for mid in module_ids:
    module_file[mid] = slug_for_section(mid, modules[mid].title, is_module=True)
  for mid in module_ids:
    for sub in subs_by_module.get(mid, []):
      # Generate one HTML per knowledge point (subsection) to satisfy the “可 reference 到所有知识点”目录诉求。
      sub_file[sub.id] = slug_for_section(sub.id, sub.title, is_module=False)

  # Build index.html
  cards: List[str] = []
  for mid in module_ids:
    m = modules[mid]
    subs = subs_by_module.get(mid, [])
    sub_links = []
    for sub in subs:
      fn = sub_file.get(sub.id)
      if fn:
        sub_links.append(f'<li><a href="./{escape(fn)}">{escape(sub.id)} {escape(sub.title)}</a></li>')
    cards.append(f"""
<div class="card">
  <h2><a href="./{escape(module_file[mid])}">{escape(mid)} {escape(m.title)}</a></h2>
  <p class="muted">来自手册的要点 + 例题答案，并补充公式/论文/代码解读与交互 Demo（关键主题）。</p>
  {'<ul>' + ''.join(sub_links) + '</ul>' if sub_links else ''}
</div>
""")

  index_body = f"""
<div class="hero">
  <h1>Transformer 与 LLM：交互式学习手册（站点版）</h1>
  <p>按模块拆分（0–13）。每页包含手册原文要点/例题与答案，并对关键算法补充公式、代码逐段解读、以及原论文与典型代码库引用。</p>
  <div class="pillrow">
    <span class="pill">MathJax 公式</span>
    <span class="pill">代码高亮</span>
    <span class="pill">Attention/PE 等可交互 Demo</span>
    <span class="pill">论文 & Codebase 回链</span>
  </div>
</div>

<div class="grid">
  {''.join(cards)}
</div>

<div class="section">
  <h2>怎么用这套站点学习</h2>
  <ul>
    <li>先看 <b>模块页</b> 抓主线，再点进 <b>知识点页</b> 做公式与交互练习。</li>
    <li>读到算法时，优先对照 “原论文与代码库” 小节，建立 <b>公式 → 代码 → 工程约束</b> 的映射。</li>
    <li>遇到概念不稳，回到模块页的 “例题与答案” 做自测。</li>
  </ul>
</div>
"""
  write(OUT_DIR / "index.html", page_template(title="目录｜Transformer × LLM 学习站", subtitle="目录与模块导航", body_html=index_body, nav_active="index"))

  # Helper: determine nav active
  def nav_for(mid: str) -> str:
    if mid == "2": return "m2"
    if mid == "7": return "m7"
    if mid == "8": return "m8"
    return ""

  # Build module pages
  all_pages_in_order: List[Tuple[str, str, str]] = []  # (key, filename, title)
  for mid in module_ids:
    all_pages_in_order.append((f"m{mid}", module_file[mid], f"{mid} {modules[mid].title}"))
    for sub in subs_by_module.get(mid, []):
      fn = sub_file.get(sub.id)
      if fn:
        all_pages_in_order.append((sub.id, fn, f"{sub.id} {sub.title}"))

  order_idx = {k:i for i,(k,_,_) in enumerate(all_pages_in_order)}

  def pager_for(key: str) -> Tuple[Optional[Tuple[str,str]], Optional[Tuple[str,str]]]:
    i = order_idx[key]
    prev = all_pages_in_order[i-1] if i-1 >= 0 else None
    nxt = all_pages_in_order[i+1] if i+1 < len(all_pages_in_order) else None
    prev_link = (prev[1], prev[2]) if prev else None
    next_link = (nxt[1], nxt[2]) if nxt else None
    return prev_link, next_link

  for mid in module_ids:
    m = modules[mid]
    subs = subs_by_module.get(mid, [])
    sub_cards = []
    for sub in subs:
      fn = sub_file.get(sub.id)
      if not fn:
        continue
      sub_cards.append(f"""
<div class="card third">
  <h2><a href="./{escape(fn)}">{escape(sub.id)} {escape(sub.title)}</a></h2>
  <p class="muted">知识点页（含公式/代码解读与可能的交互 demo）。</p>
</div>
""")

    body = f"""
<div class="hero">
  <h1>{escape(mid)} {escape(m.title)}</h1>
  <p class="muted">本页是模块汇总（来自 PDF 原文），并链接到本模块的重要知识点页。</p>
</div>

<div class="grid">
  {''.join(sub_cards) if sub_cards else '<div class="card full"><p class="muted">本模块暂无单独拆出的知识点页（后续可按需要继续拆分）。</p></div>'}
</div>

<div class="section">
  <h2>手册原文（整理版）</h2>
  {section_to_html(m.lines)}
</div>
"""
    write(OUT_DIR / module_file[mid], page_template(
      title=f"{mid} {m.title}｜Transformer × LLM 学习站",
      subtitle=f"模块 {mid}",
      body_html=body,
      nav_active=nav_for(mid),
      pager=pager_for(f"m{mid}")
    ))

  # Build knowledge point pages (selected)
  for mid in module_ids:
    for sub in subs_by_module.get(mid, []):
      fn = sub_file.get(sub.id)
      if not fn:
        continue
      extra = extras.get(sub.id, "")
      body = f"""
<div class="hero">
  <h1>{escape(sub.id)} {escape(sub.title)}</h1>
  <p class="muted">先读手册要点，再看公式/代码/交互 demo 与论文/代码库引用。</p>
  <div class="pillrow">
    <span class="pill">模块 {escape(mid)}</span>
    <span class="pill"><a href="./{escape(module_file[mid])}">返回模块页</a></span>
  </div>
</div>

<div class="section">
  <h2>手册要点 + 例题答案（原文整理）</h2>
  {section_to_html(sub.lines)}
</div>

{f'<div class="section"><h2>公式 / 代码 / 交互补充（重点）</h2>{extra}</div>' if extra else ''}
"""
      write(OUT_DIR / fn, page_template(
        title=f"{sub.id} {sub.title}｜Transformer × LLM 学习站",
        subtitle=f"知识点 {sub.id}",
        body_html=body,
        nav_active=nav_for(mid),
        pager=pager_for(sub.id)
      ))


if __name__ == "__main__":
  if not PDF_PATH.exists():
    raise SystemExit(f"PDF not found: {PDF_PATH}")
  build()
  print("Built:", OUT_DIR)

