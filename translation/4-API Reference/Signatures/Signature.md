# /translation/signature.md

## dspy.Signature

```python
dspy.Signature
```

**Kullanılan Yapılar (Bases):** `BaseModel`

---

## Functions (Fonksiyonlar)

### `append`

```python
append(name, field, type_=None) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

`inputs` (girdiler) veya `outputs` (çıktılar) bölümünün sonuna bir alan (field) ekler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Eklenecek alanın adı. | **Gerekli (required)** |
| `field` | - | Eklenecek `InputField` veya `OutputField` örneği. | **Gerekli (required)** |
| `type_` | `type \| None` | İsteğe bağlı açık (explicit) tür ek açıklaması. `type_` `None` ise, geçerli (effective) tür `insert` tarafından çözümlenir. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Alanın eklendiği yeni bir Signature sınıfı. |

**Örnekler (Examples):**

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig.fields.keys()))
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `delete`

```python
delete(name) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

Verilen alan olmadan yeni bir Signature sınıfı döndürür.
Eğer `name` (isim) mevcut değilse, alanlar değişmeden kalır (hata fırlatılmaz).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Kaldırılacak alanın adı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Alanın kaldırıldığı (veya alan yoksa değişmeden kalan) yeni bir Signature sınıfı. |

**Örnekler (Examples):**

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    temp_field: str = dspy.InputField(desc="Temporary debug field")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.delete("temp_field")
print(list(NewSig.fields.keys()))

# Alan mevcut değilse hata fırlatılmaz
Unchanged = NewSig.delete("nonexistent")
print(list(Unchanged.fields.keys()))
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `dump_state`

```python
dump_state()
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `equals`

```python
equals(other) -> bool
```

*(Sınıf Metodu / classmethod)*

İki Signature sınıfının JSON şemasını karşılaştırır.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `insert`

```python
insert(index: int, name: str, field, type_: type | None = None) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

Girdiler veya çıktılar arasına belirli bir konuma bir alan ekler.
Negatif indeksler desteklenir (örn. `-1` sona ekler). Eğer `type_` atlanırsa, alanın mevcut `annotation`'ı (ek açıklaması) kullanılır; eğer bu da eksikse `str` kullanılır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `index` | `int` | Seçilen bölüm içindeki ekleme konumu; negatifler sona ekler. | **Gerekli (required)** |
| `name` | `str` | Eklenecek alanın adı. | **Gerekli (required)** |
| `field` | - | Eklenecek `InputField` veya `OutputField` örneği. | **Gerekli (required)** |
| `type_` | `type \| None` | İsteğe bağlı açık tür ek açıklaması. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Alanın eklendiği yeni bir Signature sınıfı. |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | `index`, seçilen bölüm için geçerli aralığın dışına düşerse. |

**Örnekler (Examples):**

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))

NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig2.fields.keys()))
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `load_state`

```python
load_state(state)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `prepend`

```python
prepend(name, field, type_=None) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

`inputs` veya `outputs` bölümünün 0. indeksine (başına) bir alan ekler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Eklenecek alanın adı. | **Gerekli (required)** |
| `field` | - | Eklenecek `InputField` veya `OutputField` örneği. | **Gerekli (required)** |
| `type_` | `type \| None` | İsteğe bağlı açık tür ek açıklaması. `type_` `None` ise, geçerli tür `insert` tarafından çözümlenir. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Alanın ilk sıraya eklendiği yeni bir `Signature` sınıfı. |

**Örnekler (Examples):**

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `with_instructions`

```python
with_instructions(instructions: str) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

Aynı alanlara ve yeni talimatlara sahip yeni bir Signature sınıfı döndürür.
Bu metot `cls`'yi (sınıfın kendisini) değiştirmez. Mevcut alanları ve sağlanan `instructions`'ı (talimatları) kullanarak yeni bir Signature sınıfı inşa eder.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `instructions` | `str` | Yeni imzaya eklenecek talimat metni. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Alanları `cls.fields` ile eşleşen ve talimatları `instructions`'a eşit olan yeni bir Signature sınıfı. |

**Örnekler (Examples):**

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input text")
    output_text: str = dspy.OutputField(desc="Output text")

NewSig = MySig.with_instructions("Translate to French.")
assert NewSig is not MySig
assert NewSig.instructions == "Translate to French."
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*

### `with_updated_fields`

```python
with_updated_fields(name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

Güncellenmiş alan bilgileriyle yeni bir Signature sınıfı oluşturur.
`fields[name].json_schema_extra[key] = value` ile güncellenen alan ve isimle birlikte yeni bir Signature sınıfı döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Güncellenecek alanın adı. | **Gerekli (required)** |
| `type_` | `type \| None` | Alanın yeni türü. | `None` |
| `kwargs` | `dict[str, Any]` | Alan için yeni değerler. | `{}` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Güncellenmiş alan bilgileriyle yeni bir Signature sınıfı (bir örnek (instance) değil). |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/signatures/signature.py`*
