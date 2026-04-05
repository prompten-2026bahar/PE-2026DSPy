# /translation/bootstrap_few_shot_with_random_search.md

## dspy.BootstrapFewShotWithRandomSearch

```python
dspy.BootstrapFewShotWithRandomSearch(metric, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=None, max_errors=None, stop_at_score=None, metric_threshold=None)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/random_search.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/random_search.py`*

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
