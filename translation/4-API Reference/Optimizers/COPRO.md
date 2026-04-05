# /translation/copro.md

## dspy.COPRO

```python
dspy.COPRO(prompt_model=None, metric=None, breadth=10, depth=3, init_temperature=1.4, track_stats=False, **_kwargs)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`



*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/copro_optimizer.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student, *, trainset, eval_kwargs)
```

Öğrenci (`student`) programının imzasını (signature) optimize eder - sıfır atışlı (zero-shot) veya halihazırda önceden optimize edilmiş (demolar zaten seçilmiş - `demos != []`) olabileceğini unutmayın.

**Parametreler:**
* `student`: Optimize edilecek ve üzerinde değişiklik yapılacak program.
* `trainset`: Yinelenebilir (iterable) `Example` (Örnek) dizisi.
* `eval_kwargs`: İsteğe bağlı, sözlük (`dict`). Metrik için `Evaluate` sınıfına aktarılacak ek anahtar kelime argümanları.

**Dönüş Değerleri:**
Öğrenci (`student`) programının optimize edilmiş sürümünü döndürür.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/copro_optimizer.py`*

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
