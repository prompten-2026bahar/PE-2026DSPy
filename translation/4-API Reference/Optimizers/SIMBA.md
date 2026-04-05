# /translation/simba.md

## dspy.SIMBA

```python
dspy.SIMBA(*, metric: Callable[[dspy.Example, dict[str, Any]], float], bsize: int = 32, num_candidates: int = 6, max_steps: int = 8, max_demos: int = 4, prompt_model: dspy.LM | None = None, teacher_settings: dict | None = None, demo_input_field_maxlen: int = 100000, num_threads: int | None = None, temperature_for_sampling: float = 0.2, temperature_for_candidates: float = 0.2)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

DSPy için SIMBA (Stochastic Introspective Mini-Batch Ascent / Stokastik İçgözlemsel Mini-Yığın Tırmanışı) optimize edicisi.

SIMBA, LLM'nin kendi performansını analiz etmesi ve iyileştirme kuralları üretmesi için LLM'yi kullanan bir DSPy optimize edicisidir. Mini yığınları (mini-batches) örnekler, yüksek çıktı değişkenliğine sahip zorlu örnekleri belirler, ardından ya kendi üzerine düşünen (self-reflective) kurallar oluşturur ya da başarılı örnekleri gösterim (demonstrations) olarak ekler.

Daha fazla detay için şu adrese bakabilirsiniz: [https://dspy.ai/api/optimizers/SIMBA/](https://dspy.ai/api/optimizers/SIMBA/)

SIMBA'yı başlatır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `metric` | `Callable[[Example, dict[str, Any]], float]` | Girdi olarak bir `Example` (Örnek) ve bir `prediction_dict` (tahmin sözlüğü) alan ve bir ondalık sayı (`float`) döndüren fonksiyon. | **Gerekli (required)** |
| `bsize` | `int` | Mini yığın (Mini-batch) boyutu. Varsayılan 32'dir. | `32` |
| `num_candidates` | `int` | Her yinelemede (iteration) üretilecek yeni aday program sayısı. Varsayılan 6'dır. | `6` |
| `max_steps` | `int` | Çalıştırılacak optimizasyon adımı sayısı. Varsayılan 8'dir. | `8` |
| `max_demos` | `int` | Bir tahmincinin bazılarını düşürmeden (dropping) önce tutabileceği maksimum demo sayısı. Varsayılan 4'tür. | `4` |
| `prompt_model` | `LM \| None` | Programı evrimleştirmek için kullanılacak model. `prompt_model` None olduğunda, global olarak yapılandırılmış lm kullanılır. | `None` |
| `teacher_settings` | `dict \| None` | Öğretmen (teacher) modeli için ayarlar. Varsayılan None'dır. | `None` |
| `demo_input_field_maxlen` | `int` | Yeni bir demo oluştururken girdi alanında tutulacak maksimum karakter sayısı. Varsayılan 100.000'dir. | `100000` |
| `num_threads` | `int \| None` | Paralel yürütme için iş parçacığı sayısı. Varsayılan None'dır. | `None` |
| `temperature_for_sampling` | `float` | Yörünge örnekleme (trajectory-sampling) adımında programları seçmek için kullanılan sıcaklık. Varsayılan 0.2'dir. | `0.2` |
| `temperature_for_candidates` | `float` | Yeni adaylar oluşturmak için kaynak programı seçerken kullanılan sıcaklık. Varsayılan 0.2'dir. | `0.2` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/simba.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student: dspy.Module, *, trainset: list[dspy.Example], seed: int = 0) -> dspy.Module
```

Öğrenci modülünü SIMBA kullanarak derleyin ve optimize edin.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `student` | `Module` | Optimize edilecek modül. | **Gerekli (required)** |
| `trainset` | `list[Example]` | Optimizasyon için eğitim örnekleri. | **Gerekli (required)** |
| `seed` | `int` | Tekrarlanabilirlik için rastgele tohum (random seed). | `0` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Module` | İliştirilmiş `candidate_programs` (aday programlar) ve `trial_logs` (deneme günlükleri) ile optimize edilmiş modül. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/simba.py`*

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

---

## Örnek Kullanım (Example Usage)

```python
optimizer = dspy.SIMBA(metric=your_metric)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Optimize edilmiş programı gelecekte kullanmak üzere kaydedin
optimized_program.save(f"optimized.json")
```

---

## SIMBA Nasıl Çalışır? (How SIMBA works)

SIMBA (Stochastic Introspective Mini-Batch Ascent / Stokastik İçgözlemsel Mini-Yığın Tırmanışı), LLM'nin kendi performansını analiz etmesi ve iyileştirme kuralları üretmesi için LLM'yi kullanan bir DSPy optimize edicisidir. Mini yığınları (mini-batches) örnekler, yüksek çıktı değişkenliğine sahip zorlu örnekleri belirler, ardından kendi üzerine düşünen (self-reflective) kurallar oluşturur veya başarılı örnekleri gösterim (demonstrations) olarak ekler. Daha fazla detay için Marius'un yazdığı bu harika blog yazısına bakabilirsiniz.
