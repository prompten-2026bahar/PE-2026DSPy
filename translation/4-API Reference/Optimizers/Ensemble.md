# /translation/ensemble.md

## dspy.Ensemble

```python
dspy.Ensemble(*, reduce_fn=None, size=None, deterministic=False)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`



Yaygın bir `reduce_fn` (indirgeme fonksiyonu) `dspy.majority`'dir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/ensemble.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(programs)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/ensemble.py`*

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
