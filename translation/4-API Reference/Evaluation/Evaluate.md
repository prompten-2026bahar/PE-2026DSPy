# /translation/evaluate.md

## dspy.Evaluate

```python
dspy.Evaluate(
    *,
    devset: list[dspy.Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    display_progress: bool = False,
    display_table: bool | int = False,
    max_errors: int | None = None,
    provide_traceback: bool | None = None,
    failure_score: float = 0.0,
    save_as_csv: str | None = None,
    save_as_json: str | None = None,
    **kwargs
)
```

DSPy Evaluate sınıfı.

Bu sınıf, bir DSPy programının performansını değerlendirmek için kullanılır. Kullanıcıların bu sınıfı kullanabilmeleri için bir değerlendirme veri kümesi (evaluation dataset) ve bir metrik fonksiyonu (metric function) sağlamaları gerekir. Bu sınıf, sağlanan veri kümesi üzerinde paralel değerlendirmeyi (parallel evaluation) destekler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `devset` | `list[Example]` | Değerlendirme veri kümesi (evaluation dataset). | **Gerekli (required)** |
| `metric` | `Callable` | Değerlendirme için kullanılacak metrik fonksiyonu. | `None` |
| `num_threads` | `Optional[int]` | Paralel değerlendirme için kullanılacak iş parçacığı sayısı. | `None` |
| `display_progress` | `bool` | Değerlendirme sırasında ilerleme durumunun gösterilip gösterilmeyeceği. | `False` |
| `display_table` | `Union[bool, int]` | Değerlendirme sonuçlarının bir tabloda gösterilip gösterilmeyeceği. Bir sayı geçilirse, değerlendirme sonuçları görüntülenmeden önce o sayıya kadar kısaltılır. | `False` |
| `max_errors` | `Optional[int]` | Değerlendirmeyi durdurmadan önce izin verilen maksimum hata sayısı. `None` ise, `dspy.settings.max_errors` değerini miras alır. | `None` |
| `provide_traceback` | `Optional[bool]` | Değerlendirme sırasında izleme (traceback) bilgisinin sağlanıp sağlanmayacağı. | `None` |
| `failure_score` | `float` | Bir istisna (exception) nedeniyle değerlendirme başarısız olursa kullanılacak varsayılan skor. | `0.0` |
| `save_as_csv` | `Optional[str]` | CSV dosyasının kaydedileceği dosya adı. | `None` |
| `save_as_json` | `Optional[str]` | JSON dosyasının kaydedileceği dosya adı. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/evaluate.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(
    program: dspy.Module,
    metric: Callable | None = None,
    devset: list[dspy.Example] | None = None,
    num_threads: int | None = None,
    display_progress: bool | None = None,
    display_table: bool | int | None = None,
    callback_metadata: dict[str, Any] | None = None,
    save_as_csv: str | None = None,
    save_as_json: str | None = None
) -> EvaluationResult
```

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `program` | `Module` | Değerlendirilecek DSPy programı. | **Gerekli (required)** |
| `metric` | `Callable` | Değerlendirme için kullanılacak metrik fonksiyonu. Sağlanmazsa `self.metric` kullanılır. | `None` |
| `devset` | `list[Example]` | Değerlendirme veri kümesi. Sağlanmazsa `self.devset` kullanılır. | `None` |
| `num_threads` | `Optional[int]` | Paralel değerlendirme için kullanılacak iş parçacığı sayısı. Sağlanmazsa `self.num_threads` kullanılır. | `None` |
| `display_progress` | `bool` | Değerlendirme sırasında ilerleme durumunun gösterilip gösterilmeyeceği. Sağlanmazsa `self.display_progress` kullanılır. | `None` |
| `display_table` | `Union[bool, int]` | Değerlendirme sonuçlarının bir tabloda gösterilip gösterilmeyeceği. Sağlanmazsa `self.display_table` kullanılır. Bir sayı geçilirse, sonuçlar o sayıya kadar kısaltılır. | `None` |
| `callback_metadata` | `dict` | Değerlendirme geri çağırma (callback) işleyicileri için kullanılacak meta veriler. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `EvaluationResult` | Değerlendirme sonuçları, aşağıdaki öznitelikleri (attributes) içeren bir `dspy.EvaluationResult` nesnesi olarak döndürülür: |
| | **score**: Genel performansı temsil eden yüzde puanı (örn. 67.30). |
| | **results**: Veri kümesindeki (devset) her bir örnek için `(example, prediction, score)` üçlülerinden oluşan bir liste. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/evaluate.py`*

