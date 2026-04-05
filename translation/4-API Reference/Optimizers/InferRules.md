# /translation/infer_rules.md

## dspy.InferRules

```python
dspy.InferRules(num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs)
```

**Kullanılan Yapılar (Bases):** `BootstrapFewShot`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student, *, teacher=None, trainset, valset=None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

### `evaluate_program`

```python
evaluate_program(program, dataset)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

### `format_examples`

```python
format_examples(demos, signature)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

### `get_params`

```python
get_params() -> dict[str, Any]
```

Teleprompter'ın parametrelerini alır.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `dict[str, Any]` | Teleprompter'ın parametreleri. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/teleprompt.py`*

### `get_predictor_demos`

```python
get_predictor_demos(trainset, predictor)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

### `induce_natural_language_rules`

```python
induce_natural_language_rules(predictor, trainset)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*

### `update_program_instructions`

```python
update_program_instructions(predictor, natural_language_rules)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/infer_rules.py`*
