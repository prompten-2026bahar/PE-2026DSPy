# /translation/parallel.md

## dspy.Parallel

```python
dspy.Parallel(num_threads: int | None = None, max_errors: int | None = None, access_examples: bool = True, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False, timeout: int = 120, straggler_limit: int = 3)
```

`(modül, örnek)` çiftlerinin paralel, çok iş parçacıklı (multi-threaded) yürütülmesi için bir yardımcı (utility) sınıftır. Çeşitli örnek formatlarını (örn. `Example`, sözlük, demet, liste) destekler, sağlam (robust) hata işleme ve isteğe bağlı ilerleme takibi sunar; ayrıca isteğe bağlı olarak başarısız örnekleri ve istisnaları (exceptions) döndürebilir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `num_threads` | `Optional[int]` | Kullanılacak iş parçacığı sayısı. Varsayılan olarak `settings.num_threads` değerini alır. | `None` |
| `max_errors` | `Optional[int]` | Bir istisna fırlatmadan önce izin verilen maksimum hata sayısı. Varsayılan olarak `settings.max_errors` değerini alır. | `None` |
| `access_examples` | `bool` | `Example` nesnelerinin `.inputs()` aracılığıyla paketinden çıkarılıp çıkarılmayacağı (unpack). | `True` |
| `return_failed_examples` | `bool` | Başarısız örneklerin döndürülüp döndürülmeyeceği. | `False` |
| `provide_traceback` | `Optional[bool]` | İzleme (traceback) bilgisinin sağlanıp sağlanmayacağı. | `None` |
| `disable_progress_bar` | `bool` | İlerleme çubuğunun devre dışı bırakılıp bırakılmayacağı. | `False` |
| `timeout` | `int` | Geride kalan (straggler) bir görevin yeniden gönderilmesinden önceki saniye süresi. Devre dışı bırakmak için 0 yapın. | `120` |
| `straggler_limit` | `int` | Yalnızca bu kadar veya daha az görev kaldığında geride kalanları kontrol eder. | `3` |

**Örnek (Example):**

```python
import dspy
from dspy import Parallel

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

examples = [
    {"question": "What is the capital of Spain?"},
    {"question": "What is 3 * 4?"},
    {"question": "Who wrote Hamlet?"},
]

module = dspy.Predict("question->answer")
exec_pairs = [(module, example) for example in examples]

parallel = Parallel(num_threads=3, disable_progress_bar=False)
results = parallel(exec_pairs)

for i, result in enumerate(results):
    print(f"Result {i+1}: {result.answer}")

# Beklenen Çıktı (Expected Output):
# Result 1: Madrid
# Result 2: 12
# Result 3: William Shakespeare
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/parallel.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(*args: Any, **kwargs: Any) -> Any
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/parallel.py`*

### `forward`

```python
forward(exec_pairs: list[tuple[Any, Example]], num_threads: int | None = None) -> list[Any]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/parallel.py`*
