# /translation/prediction.md

## dspy.Prediction

```python
dspy.Prediction(*args, **kwargs)
```

**Kullanılan Yapılar (Bases):** `Example`

Bir DSPy modülünün çıktısını içeren bir tahmin (prediction) nesnesi.

`Prediction`, `Example` sınıfından miras alır.

Geri bildirimle artırılmış skorlara (feedback-augmented scores) olanak tanımak için `Prediction`, bir `score` (skor) alanına sahip Tahminler (Predictions) için karşılaştırma işlemlerini (`<`, `>`, `<=`, `>=`) destekler. Karşılaştırma işlemleri 'score' değerlerini ondalık sayı (`float`) olarak karşılaştırır. Eşitlik karşılaştırması için, altta yatan (underlying) veri depoları eşitse Tahminler de eşittir (`Example` sınıfından miras alınmıştır).

'score' alanına sahip Tahminler için aritmetik işlemler (`+`, `/`, vb.) de desteklenir ve doğrudan skor değeri üzerinde işlem yapar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`*

---

## Functions (Fonksiyonlar)

### `copy`

```python
copy(**kwargs)
```

İsteğe bağlı olarak alanları geçersiz kılarak (overriding) yüzeysel bir kopya (shallow copy) döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `**kwargs` | - | Kopyada eklenecek veya geçersiz kılınacak alanlar. | `{}` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.")
>>> ex.copy(answer="No reason.")
Example({'question': 'Why?', 'answer': 'No reason.'}) (input_keys=None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `from_completions`

```python
from_completions(list_or_dict, signature=None)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`*

### `get`

```python
get(key, default=None)
```

`key` (anahtar) için değeri döndürür veya alan mevcut değilse `default` (varsayılan) değeri döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `key` | - | Aranacak alan adı. | **Gerekli (required)** |
| `default` | - | `key` eksik olduğunda döndürülecek değer. | `None` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(name="Alice")
>>> ex.get("name")
'Alice'
>>> ex.get("city", "Unknown")
'Unknown'
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `get_lm_usage`

```python
get_lm_usage()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`*

### `inputs`

```python
inputs()
```

Yalnızca girdi (input) alanlarını içeren yeni bir `Example` döndürür.
Önceden `with_inputs`'un çağrılmış olmasını gerektirir.

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | Bu örnekte `with_inputs` çağrılmamışsa. |

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.inputs()
Example({'question': 'Why?'}) (input_keys={'question'})
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `items`

```python
items(include_dspy=False)
```

`dict.items()` gibi `(field_name, value)` yani `(alan_adı, değer)` çiftlerini döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `include_dspy` | - | `True` ise, `dspy_` önekiyle başlayan dahili (internal) alanları dahil eder. | `False` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").items()
[('question', 'Why?'), ('answer', 'Because.')]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `keys`

```python
keys(include_dspy=False)
```

`dict.keys()` gibi alan adlarını döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `include_dspy` | - | `True` ise, `dspy_` önekiyle başlayan dahili alanları dahil eder. Normalde bunları görmezden gelebilirsiniz. | `False` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").keys()
['question', 'answer']
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `labels`

```python
labels()
```

Yalnızca etiket (label - girdi olmayan) alanlarını içeren yeni bir `Example` döndürür.
Etiketler girdi *olmayan* her şey olduğu için önceden `with_inputs`'un çağrılmış olmasını gerektirir.

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.labels()
Example({'answer': 'Because.'}) (input_keys=None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `set_lm_usage`

```python
set_lm_usage(value)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`*

### `toDict`

```python
toDict()
```

İç içe geçmiş nesneleri özyineli (recursively) olarak serileştirerek düz bir sözlüğe dönüştürür.
İç içe geçmiş `Example` nesneleri, Pydantic modelleri, listeler ve sözlükler dönüştürülür, böylece sonuç JSON uyumlu olur.

**Örnekler (Examples):**
```python
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").toDict()
{'question': 'Why?', 'answer': 'Because.'}
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `values`

```python
values(include_dspy=False)
```

`dict.values()` gibi alan değerlerini döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `include_dspy` | - | `True` ise, `dspy_` önekiyle başlayan dahili alanları dahil eder. | `False` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").values()
['Why?', 'Because.']
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `with_inputs`

```python
with_inputs(*keys)
```

Hangi alanların girdi (input) olduğunu işaretler ve yeni bir `Example` döndürür.
Burada listelenmeyen alanlar etiketler (labels - beklenen çıktılar) olarak ele alınır. DSPy optimize edicileri ve değerlendiricileri (evaluators) bu ayrımı kullanır: programınıza `example.inputs()` öğesini iletirler ve çıktıyı `example.labels()` ile karşılaştırırlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `*keys` | - | Girdi alanlarının adları. | `()` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| - | Girdi anahtarları (input keys) ayarlanmış olarak bu `Example`'ın bir kopyası. |

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.inputs().keys()
['question']
>>> ex.labels().keys()
['answer']
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

### `without`

```python
without(*keys)
```

Belirtilen alanların kaldırıldığı bir kopya döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `*keys` | - | Düşürülecek (kaldırılacak) alan adları. | `()` |

**Örnekler (Examples):**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.", source="web")
>>> ex.without("source")
Example({'question': 'Why?', 'answer': 'Because.'}) (input_keys=None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*
