# /translation/answer_passage_match.md

## dspy.evaluate.answer_passage_match

```python
dspy.evaluate.answer_passage_match(example, pred, trace=None)
```

`pred.context` içindeki herhangi bir pasaj (passage) yanıtı/yanıtları içeriyorsa `True` döndürür.

String'ler (dizgiler) normalleştirilir (ve pasajlar da dahili olarak DPR normalizasyonu kullanır).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `example` | - | `answer` (`str` veya `list[str]`) alanına sahip `dspy.Example` nesnesi. | **Gerekli (required)** |
| `pred` | - | Pasajları barındıran `context` (`list[str]`) alanına sahip `dspy.Prediction` nesnesi. | **Gerekli (required)** |
| `trace` | - | Kullanılmaz; uyumluluk (compatibility) için ayrılmıştır. | `None` |

**Dönüş Değerleri (Returns):**

| İsim (Name) | Tip (Type) | Açıklama (Description) |
| :--- | :--- | :--- |
| - | `bool` | Herhangi bir pasaj referans yanıtlardan herhangi birini içeriyorsa `True`; aksi takdirde `False`. |

**Örnekler (Examples):**

```python
import dspy

example = dspy.Example(answer="Eiffel Tower")
pred = dspy.Prediction(context=["The Eiffel Tower is in Paris.", "..."])

answer_passage_match(example, pred)  # True
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/metrics.py`*

