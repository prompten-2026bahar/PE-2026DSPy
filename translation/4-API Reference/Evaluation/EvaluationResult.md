# /translation/evaluation_result.md

## dspy.evaluate.EvaluationResult

```python
dspy.evaluate.EvaluationResult(score: float, results: list[tuple[dspy.Example, dspy.Example, Any]])
```

[cite_start]**Kullanılan Yapılar (Bases):** `Prediction` [cite: 3]

Bir değerlendirmenin sonucunu temsil eden sınıftır. [cite_start]`dspy.Prediction` sınıfının bir alt sınıfıdır ve aşağıdaki alanları içerir: [cite: 3]

* [cite_start]**score**: Genel performansı temsil eden bir float değeri (örn. 67.30). [cite: 3]
* [cite_start]**results**: Veri kümesindeki (devset) her bir örnek için `(example, prediction, score)` üçlülerinden (tuples) oluşan bir liste. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/evaluate.py`* [cite: 3]

---

## Functions (Fonksiyonlar)

### `copy`

```python
copy(**kwargs)
```

[cite_start]Alanları isteğe bağlı olarak geçersiz kılarak (overriding) yüzeysel bir kopya (shallow copy) döndürür. [cite: 3]

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `**kwargs` | - | Kopyada eklenecek veya geçersiz kılınacak alanlar. | [cite_start]`{}` [cite: 3] |

**Örnekler:**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.")
>>> ex.copy(answer="No reason.")
Example({'question': 'Why?', 'answer': 'No reason.'}) (input_keys=None)
```
[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `from_completions`

```python
from_completions(list_or_dict, signature=None) # classmethod
```
[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`* [cite: 3]

### `get`

```python
get(key, default=None)
```

[cite_start]Belirtilen anahtar (`key`) için değeri döndürür; alan mevcut değilse `default` değerini döndürür. [cite: 3]

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `key` | - | Bakılacak alan adı. | [cite_start]**Gerekli (required)** [cite: 3] |
| `default` | - | Anahtar eksik olduğunda döndürülecek değer. | [cite_start]`None` [cite: 3] |

**Örnekler:**
```python
>>> import dspy
>>> ex = dspy.Example(name="Alice")
>>> ex.get("name")
'Alice'
>>> ex.get("city", "Unknown")
'Unknown'
```
[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `get_lm_usage`

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`* [cite: 3]

### `inputs`

[cite_start]Yalnızca girdi alanlarını (input fields) içeren yeni bir `Example` döndürür. [cite: 3] [cite_start]Öncelikle `with_inputs` metodunun çağrılmış olmasını gerektirir. [cite: 3]

**Hatalar (Raises):**
* **ValueError**: Eğer bu örnek üzerinde `with_inputs` çağrılmamışsa. [cite: 3]

**Örnekler:**
```python
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.inputs()
Example({'question': 'Why?'}) (input_keys={'question'})
```
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `items`

```python
items(include_dspy=False)
```

`dict.items()` gibi `(field_name, value)` çiftlerini döndürür. [cite: 3]

**Parametreler:**
* [cite_start]**include_dspy** (`bool`): `True` ise, `dspy_` önekiyle başlayan dahili alanları dahil eder. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `keys`

```python
keys(include_dspy=False)
```

[cite_start]`dict.keys()` gibi alan adlarını döndürür. [cite: 3]

**Parametreler:**
* **include_dspy** (`bool`): `True` ise, `dspy_` önekiyle başlayan dahili alanları dahil eder. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `labels`

[cite_start]Yalnızca etiket (label - girdi olmayan) alanlarını içeren yeni bir `Example` döndürür. [cite: 3] Girdilerin ne olduğunu belirlemek için `with_inputs` çağrılmış olmalıdır; [cite_start]çünkü etiketler, girdi olmayan her şeydir. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `set_lm_usage`

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/prediction.py`* [cite: 3]

### `toDict`

[cite_start]İç içe geçmiş nesneleri yinelemeli olarak serileştirerek düz bir sözlüğe dönüştürür. [cite: 3] [cite_start]JSON dostu bir sonuç için iç içe geçmiş `Example` nesneleri, Pydantic modelleri, listeler ve sözlükler dönüştürülür. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `values`

```python
values(include_dspy=False)
```

[cite_start]`dict.values()` gibi alan değerlerini döndürür. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `with_inputs`

```python
with_inputs(*keys)
```

[cite_start]Hangi alanların girdi (input) olduğunu işaretler ve yeni bir `Example` döndürür. [cite: 3] [cite_start]Burada listelenmeyen alanlar etiket (expected outputs) olarak kabul edilir. [cite: 3] [cite_start]DSPy optimize edicileri ve değerlendiricileri bu ayrımı kullanır: `example.inputs()` kısmını programınıza iletir ve çıktıyı `example.labels()` ile karşılaştırırlar. [cite: 3]

**Parametreler:**
* **\*keys**: Girdi alanlarının isimleri. [cite: 3]

**Dönüş Değerleri (Returns):**
* Girdi anahtarları ayarlanmış bu `Example` nesnesinin bir kopyası. [cite: 3]

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

### `without`

```python
without(*keys)
```

[cite_start]Belirtilen alanların kaldırıldığı bir kopya döndürür. [cite: 3]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/example.py`* [cite: 3]

