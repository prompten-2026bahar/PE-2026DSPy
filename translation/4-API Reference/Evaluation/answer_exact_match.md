# /translation/answer_exact_match.md

## dspy.evaluate.answer_exact_match

```python
dspy.evaluate.answer_exact_match(example, pred, trace=None, frac=1.0)
```

Bir örnek/tahmin (example/prediction) çifti için kelimesi kelimesine eşleşmeyi (exact match) veya F1 eşik değerli (F1-thresholded) eşleşmeyi değerlendirir.

Eğer `example.answer` bir string (dizgi) ise, `pred.answer` değerini onunla karşılaştırır. Eğer bir liste ise, referanslardan herhangi biriyle karşılaştırır. `frac >= 1.0` (varsayılan) olduğunda EM (Exact Match - Kesin Eşleşme) kullanır; aksi takdirde referanslar arasındaki maksimum F1 değerinin en az `frac` kadar olmasını gerektirir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `example` | - | `answer` (`str` veya `list[str]`) alanına sahip `dspy.Example` nesnesi. | **Gerekli (required)** |
| `pred` | - | `answer` (`str`) alanına sahip `dspy.Prediction` nesnesi. | **Gerekli (required)** |
| `trace` | - | Kullanılmaz; uyumluluk (compatibility) için ayrılmıştır. | `None` |
| `frac` | `float` | `[0.0, 1.0]` aralığında eşik (threshold) değeri. `1.0` değeri EM (Exact Match) anlamına gelir. | `1.0` |

**Dönüş Değerleri (Returns):**

| İsim (Name) | Tip (Type) | Açıklama (Description) |
| :--- | :--- | :--- |
| - | `bool` | Eşleşme koşulu sağlanıyorsa `True`; aksi takdirde `False`. |

**Örnekler (Examples):**

```python
import dspy

example = dspy.Example(answer=["Eiffel Tower", "Louvre"])
pred = dspy.Prediction(answer="The Eiffel Tower")

answer_exact_match(example, pred, frac=1.0)  # EM'ye (Exact Match) eşdeğer, True
answer_exact_match(example, pred, frac=0.5)  # True
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/metrics.py`*

