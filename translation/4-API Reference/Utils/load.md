# /translation/load.md

## dspy.load

```python
dspy.load(path: str, allow_pickle: bool = False) -> Module
```

Kaydedilmiş DSPy modelini yükler.
Bu metot, `save_program=True` ile kaydedilmiş, yani modelin `cloudpickle` ile kaydedildiği bir DSPy modelini yüklemek için kullanılır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | Kaydedilen modelin yolu (path). | **Gerekli (required)** |
| `allow_pickle` | `bool` | Modelin `pickle` ile yüklenmesine izin verilip verilmeyeceği. Bu tehlikelidir ve yalnızca modelin kaynağına güvendiğinizden eminseniz kullanılmalıdır. | `False` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Module` | Yüklenen model, bir `dspy.Module` örneği (instance). |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/utils/saving.py`*
