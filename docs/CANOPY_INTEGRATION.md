# Actual Canopy-Jev integration

Source: [Camellia86/Canopy-Jev](https://github.com/Camellia86/Canopy-Jev), pinned at
[`b71f97b3bffd28c90306bfea607d2089d75e7d43`](https://github.com/Camellia86/Canopy-Jev/tree/b71f97b3bffd28c90306bfea607d2089d75e7d43).
This project depends on the original package; it does not reimplement its model.

The checked path is `CanopyJev.from_pretrained` → `predict(state, questions)` →
`prompt.prepare` → `infer_tokens` → shared-prefix tree stem and adapted tail →
numerical prior. Choice supplies 1–10 candidates. Score is an ordered candidate
adaptation of the same Choice head. There is no generated reasoning/answer
token in this API. A probability is a decision preference, not a hidden tensor,
expert activation score learned on this host model, or calibrated NLL gain.

The frozen base is `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; adapter/prior artifacts are fetched
separately from `Camellia86/Canopy-Jev-27B`. The supported upstream runtime needs
Linux/NVIDIA CUDA and at least 64 GiB free GPU memory. Follow upstream pins for
PyTorch, Transformers, Triton and FLA; the generic optional PyTorch executor
extra in this project does not establish upstream full-model compatibility.

`CanopyBackend` uses the actual tokenizer and `prepare` to measure packed tokens
before each real call. It divides a large set of questions into smaller batches
without shortening the state. A single state+question larger than 16,384 packed
tokens fails. Splitting requests repeats the prefix work; it is not a persistent
cross-request KV cache. Candidate menus above ten are rejected rather than
silently shortlisted.

For each routed target, the serializer provides every permitted token with
positions, current depth, JSON execution observations, fixed expert capability
cards, and time-stamped history/metadata. Dense host hidden states and KV arrays
are not automatically provided: the host must supply a meaningful observation
interface. Full text access does not prove the resulting representation is a
sufficient statistic for optimal routing. Unvalidated expert descriptions do
not establish actual expert skills.

Autoregressive `causal` routing at position *t* reads positions 0…*t* and predicts
subsequent output. Events visible only after *t* are removed. `bidirectional`
mode explicitly permits the entire supplied input; it is appropriate for an
encoder or computation on a known prompt before generating its answer. It must
not be used to route earlier training positions from a full teacher-forced LM
sequence. Global task instructions and expert cards must be fixed ex ante;
the serializer cannot detect a future answer hidden inside arbitrary prose.

The upstream code is MIT and base-derived weights are Apache-2.0. No upstream
weights or private experiment files are redistributed here. `ReplayBackend`
contains only caller-supplied probability fixtures for CPU protocol/execution
checks and is never described as running Canopy.
