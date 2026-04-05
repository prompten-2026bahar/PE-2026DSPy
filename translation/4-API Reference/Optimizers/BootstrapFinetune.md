# /translation/bootstrap_finetune.md

## dspy.BootstrapFinetune

```python
dspy.BootstrapFinetune(metric: Callable | None = None, multitask: bool = True, train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None, adapter: Adapter | dict[LM, Adapter] | None = None, exclude_demos: bool = False, num_threads: int | None = None)
```

**Kullanılan Yapılar (Bases):** `FinetuneTeleprompter`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap_finetune.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None) -> Module
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap_finetune.py`*

### `convert_to_lm_dict`

```python
convert_to_lm_dict(arg) -> dict[LM, Any]
```

*(Statik Metot / staticmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap_finetune.py`*

### `finetune_lms`

```python
finetune_lms(finetune_dict) -> dict[Any, LM]
```

*(Statik Metot / staticmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap_finetune.py`*

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
