# /translation/example.md

## dspy.Example

```python
dspy.Example(base=None, **kwargs)
```

DSPy örnekleri (examples) ve isimlendirilmiş alanlara (named fields) sahip eğitim verileri için esnek bir veri konteyneridir (data container).

Bir `Example`, kabaca bir HuggingFace veri setinden veya pandas `DataFrame`'den alınmış bir satıra benzer. Bir sözlük (dictionary) veya nokta erişimli (dot-access) kayıt gibi davranır: alanları `example["question"]` veya `example.question` ile okuyabilirsiniz.

DSPy'da, `Example` nesnelerinden oluşan listeler sizin eğitim setiniz (trainset), geliştirme setiniz (devset) ve test setinizdir (testset). Çoğu örnek, anahtar kelime argümanlarından (keyword arguments) veya mevcut bir kayıttan oluşturulur, ardından hangi alanların bir modüle besleneceğini (fed into) belirtmek için `with_inputs(...)` ile etiketlenir. Kalan alanlar etiketler (labels) veya meta verilerdir (metadata).

Değerlendirme (evaluation) kodu, özel optimize ediciler veya eğitim döngüleri yazarken, bir modüle aktarmak istediğiniz alanlar için `example.inputs()` öğesini ve modülün çıktısıyla karşılaştırmak istediğiniz alanlar için `example.labels()` öğesini kullanın.



**Örnekler (Examples):**

Anahtar kelime argümanlarından bir tane oluşturun:
```python
>>> import dspy
>>> example = dspy.Example(
...     question="What is the capital of France?",
...     answer="Paris",
... ).with_inputs("question")
>>> example.question
'What is the capital of France?'
>>> example.answer
'Paris'
>>> example.inputs().toDict()
{'question': 'What is the capital of France?'}
```

Mevcut bir kayıttan bir tane oluşturun:
```python
>>> record = {"question": "What is 2+2?", "answer": "4"}
>>> example = dspy.Example(**record).with_inputs("question")
>>> example["question"]
'What is 2+2?'
>>> example.labels().answer
'4'
```

Hangi alanların girdi (input) olduğunu işaretleyin:
```python
>>> example = dspy.Example(
...     question="What is the weather?",
...     answer="It's sunny",
... ).with_inputs("question")
>>> example.inputs().question
'What is the weather?'
>>> example.labels().answer
"It's sunny"
```

Eğitim setinde (trainset) örnekleri kullanın:
```python
>>> trainset = [
...     dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
...     dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
... ]
>>> trainset[0].inputs().toDict()
{'question': 'What is 2+2?'}
```

Bir metrikte bir örnek kullanın:
```python
>>> def exact_match_metric(example, pred, trace=None):
...     return example.answer.lower() == pred.answer.lower()
>>> gold = dspy.Example(question="What is 1+1?", answer="2").with_inputs("question")
>>> pred = dspy.Prediction(answer="2")
>>> exact_match_metric(gold, pred)
True
```

Sözlük gibi kullanın:
```python
>>> example = dspy.Example(name="Alice", age=30).with_inputs("name")
>>> "name" in example
True
>>> example.get("city", "Unknown")
'Unknown'
```

**Ayrıca Bakınız (See Also)**
* `dspy.Evaluate`: Bir programı `Example`'lardan oluşan bir liste üzerinde değerlendirin.
* Metrikler (Metrics): Bir `Example`'ı (örnek) bir tahmin (prediction) ile karşılaştıran metrik fonksiyonları yazın.

Alanlardan veya mevcut bir kayıttan bir `Example` oluşturun.

Yaygın durumda, alanları `dspy.Example(question="...", answer="...")` gibi anahtar kelime argümanları olarak iletin. Halihazırda bir sözlüğünüz veya başka bir `Example` nesneniz olduğunda ve birkaç değeri eklemeden veya geçersiz kılmadan (overriding) önce alanlarını kopyalamak istediğinizde `base` parametresini kullanın.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `base` | - | `**kwargs` uygulanmadan önce alanların (fields) kopyalanacağı bir sözlük veya `Example`. `None` olduğunda, hiçbir alan olmadan başlar. | `None` |
| `**kwargs` | - | Örnekte (example) saklanacak alan adları ve değerleri. Bir alan hem `base` hem de `**kwargs` içinde görünürse, `**kwargs`'tan gelen değer kazanır (geçerli olur). | `{}` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`*

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
