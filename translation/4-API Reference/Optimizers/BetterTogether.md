# /translation/bettertogether.md

## dspy.BetterTogether

**BetterTogether**, Dilara Soylu, Christopher Potts ve Omar Khattab tarafından yazılan "Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together" adlı makalede önerilen bir meta-optimize edicidir. İstem optimizasyonunu (prompt optimization) ve ağırlık optimizasyonunu (weight optimization / fine-tuning) yapılandırılabilir bir sırayla uygulayarak birleştirir; bu da bir öğrenci (student) programının hem istemlerini hem de model parametrelerini yinelemeli (iterative) olarak iyileştirmesine olanak tanır.



Buradaki temel içgörü, istem ve ağırlık optimizasyonunun birbirini tamamlayabileceğidir: istem optimizasyonu potansiyel olarak etkili görev ayrıştırma (task decomposition) ve akıl yürütme stratejilerini keşfedebilirken, ağırlık optimizasyonu modeli bu kalıpları daha verimli bir şekilde yürütmek üzere uzmanlaştırabilir. Bu yaklaşımları ardışık diziler halinde (örneğin, önce istem optimizasyonu, ardından ağırlık optimizasyonu) birlikte kullanmak, her birinin diğerinin yaptığı iyileştirmeler üzerine inşa edilmesine olanak tanıyabilir.

```python
dspy.BetterTogether(metric: Callable, **optimizers: Teleprompter)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

İstem ve ağırlık optimizasyonunu yapılandırılabilir diziler (sequences) halinde birleştiren bir meta-optimize edici.

BetterTogether, "Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together" makalesinde önerilen bir meta-optimize edicidir. İstem optimizasyonunu ve ağırlık optimizasyonunu (ince ayar) yapılandırılabilir bir sırayla uygulayarak birleştirir ve bir öğrenci programının hem istemlerini hem de model parametrelerini yinelemeli olarak iyileştirmesine izin verir.

Temel içgörü, istem ve ağırlık optimizasyonunun birbirini tamamlayabileceğidir: istem optimizasyonu etkili görev ayrıştırma ve akıl yürütme stratejilerini keşfedebilirken, ağırlık optimizasyonu modeli bu kalıpları daha verimli yürütmek için uzmanlaştırabilir. Bu yaklaşımları diziler halinde (örn. önce istem, sonra ağırlık optimizasyonu) birlikte kullanmak, her birinin diğerinin yaptığı iyileştirmeler üzerine inşa edilmesini sağlayabilir. Ampirik olarak (deneysel açıdan), bu yaklaşım genellikle en son teknoloji (state-of-the-art) optimize edicilerle bile tek başına bir stratejiden daha iyi performans gösterir. Örneğin, bir Databricks vaka çalışması (case study), BetterTogether'ı GEPA ve ince ayar (fine-tuning) ile birleştirmenin, her iki yaklaşımdan tek başına daha iyi performans gösterdiğini ortaya koymuştur.

Optimize edici, bir metrik ve özel optimize ediciler ile başlatılır. Örneğin, istem optimizasyonu için GEPA'yı ağırlık optimizasyonu için BootstrapFinetune ile birleştirebilirsiniz: `BetterTogether(metric=metric, p=GEPA(...), w=BootstrapFinetune(...))`. `compile()` metodu bir öğrenci programı, eğitim seti (trainset) ve strateji anahtarlarının (strategy keys) başlatma sırasında (initialization) verilen optimize edici isimlerine karşılık geldiği bir strateji string'i alır. Her optimize ediciyi belirtilen sırada yürütür. Bir doğrulama seti (validation set) sağlandığında, en iyi performans gösteren program döndürülür; aksi takdirde en son program döndürülür.

**Not:** BootstrapFinetune gibi ağırlık optimize ediciler, öğrenci programlarının dil modellerinin açıkça ayarlanmasını gerektirir (global `dspy.settings.lm`'ye güvenmez) ve BetterTogether basitlik için bu gereksinimi yansıtır. Bu nedenle derlemeden (compile) önce `set_lm` çağırırız.

```python
>>> from dspy.teleprompt import GEPA, BootstrapFinetune
>>>
>>> # İstem optimizasyonu için GEPA'yı, ağırlık optimizasyonu için BootstrapFinetune'u birleştirin
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric, auto="medium"),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w"
... )
```

`optimizer_compile_args` kullanarak her optimize edicinin `compile()` metoduna optimize ediciye özgü (optimizer-specific) argümanlar aktarabilirsiniz. Bu, her optimize edicinin davranışını özelleştirmenizi sağlar:

```python
>>> from dspy.teleprompt import MIPROv2
>>>
>>> # Özel parametrelerle istem optimizasyonu için MIPROv2'yi kullanın
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=MIPROv2(metric=metric),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w",
...     optimizer_compile_args={
...         "p": {"num_trials": 10, "max_bootstrapped_demos": 8},  # MIPROv2'nin compile argümanlarını yapılandırın
...     }
... )
```

BetterTogether, rastgele (arbitrary) optimize edicileri sırayla çalıştırabilen bir meta-optimize edici olduğundan, optimize edicilerin herhangi bir dizisi birlikte birleştirilebilir. Strateji string'inde kullanılan optimize edici adları, kurucuda (constructor) belirtilen anahtar kelime argümanlarına (keyword arguments) karşılık gelir. Örneğin, farklı istem optimize ediciler birden çok kez dönüşümlü olarak kullanılabilir (ancak bunun BetterTogether'ın esnekliğinin bir gösterimi olduğunu, önerilen bir yapılandırma olmadığını unutmayın):

```python
>>> from dspy.teleprompt import MIPROv2, GEPA
>>>
>>> # İki optimize ediciyi üç kez zincirleyin: MIPROv2 -> GEPA -> MIPROv2
>>> optimizer = BetterTogether(
...     metric=metric,
...     mipro=MIPROv2(metric=metric, auto="light"),
...     gepa=GEPA(metric=metric, auto="light")
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="mipro -> gepa -> mipro"
... )
```

**Not (Note):**
* **Çıktı Öznitelikleri (Output Attributes):** Döndürülen program iki ek öznitelik içerir: `candidate_programs` ve `flag_compilation_error_occurred`. `candidate_programs` özniteliği, her biri 'program', 'score' ve 'strategy' (örn. '', 'p', 'p -> w', 'p -> w -> p') içeren ve skora göre azalan şekilde (descending) sıralanmış bir sözlük (dict) listesidir (`dspy.MIPROv2.candidate_programs`'a benzer şekilde). Herhangi bir optimizasyon adımı başarısız olursa, `flag_compilation_error_occurred` True olarak ayarlanır ve o ana kadar bulunan en iyi program döndürülür.
* **Model Yaşam Döngüsü Yönetimi (Model Lifecycle Management):** BetterTogether, dil modelinin yaşam döngüsünü otomatik olarak yönetir (başlatma, öldürme ve ince ayardan sonra yeniden başlatma); bunlar API tabanlı LLM'ler için etkisiz işlemlerdir (no-ops). Bu, yerel sağlayıcılarla (örn. `dspy.LocalProvider`) `BootstrapFinetune` gibi ağırlık optimize ediciler kullanırken özellikle önemlidir, çünkü optimizasyon adımları arasında modelin başlatılmasını ve temizlenmesini (cleanup) halleder.

BetterTogether'ı bir metrik ve özel optimize ediciler ile başlatın.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `metric` | `Callable` | Programları puanlamak için değerlendirme metrik fonksiyonu. `(example, prediction, trace=None)` almalı ve sayısal bir puan döndürmelidir (yüksek olan daha iyidir). Bu metrik, optimizasyon sırasında aday programları değerlendirmek için kullanılır ve özel optimize ediciler sağlanmazsa varsayılan optimize edicilere aktarılır. | **Gerekli (required)** |
| `**optimizers` | `Teleprompter` | Anahtar kelime argümanları olarak özel optimize ediciler; buradaki anahtarlar (keys), strateji string'inde kullanılan optimize edici adları haline gelir. Örneğin, `p=GEPA(...), w=BootstrapFinetune(...)` `"p -> w"` gibi stratejilerde kullanılmak üzere 'p' ve 'w'yi kullanılabilir hale getirir. Sağlanmazsa, varsayılan olarak `p=BootstrapFewShotWithRandomSearch(metric=metric)` ve `w=BootstrapFinetune(metric=metric)` değerlerini alır. Herhangi bir DSPy Teleprompter kullanılabilir. | `{}` |

**Örnekler (Examples):**

```python
>>> # Özel optimize ediciler kullanın
>>> from dspy.teleprompt import GEPA, BootstrapFinetune
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric, auto="medium"),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> # Varsayılan optimize edicileri kullanın
>>> optimizer = BetterTogether(metric=metric)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bettertogether.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student: Module, *, trainset: list[Example], teacher: Module | list[Module] | None = None, valset: list[Example] | None = None, num_threads: int | None = None, max_errors: int | None = None, provide_traceback: bool | None = None, seed: int | None = None, valset_ratio: float = 0.1, shuffle_trainset_between_steps: bool = True, strategy: str = 'p -> w -> p', optimizer_compile_args: dict[str, dict[str, Any]] | None = None) -> Module
```

Bir optimizasyon stratejileri dizisi kullanarak bir öğrenci programını derler (compile) ve optimize eder.
Strateji string'inde belirtilen optimize edicileri sırayla yürütür, her ara sonucu değerlendirir ve en iyi performans gösteren programı döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `student` | `Module` | Optimize edilecek DSPy programı. Tüm tahmincilerin atanmış dil modelleri olmalıdır. Bir programın tüm modüllerine bir dil modeli atamak için `program.set_lm(lm)` kullanılabilir. | **Gerekli (required)** |
| `trainset` | `list[Example]` | Optimizasyon için eğitim örnekleri. Her optimize edici tam `trainset`'i (veya `shuffle_trainset_between_steps=True` ise karıştırılmış bir versiyonunu) alır. | **Gerekli (required)** |
| `teacher` | `Module \| list[Module] \| None` | Önyükleme (bootstrapping) için isteğe bağlı öğretmen modülü/modülleri. Tek bir modül veya liste olabilir. Optimize edicilere aktarılır. | `None` |
| `valset` | `list[Example] \| None` | Optimizasyon adımlarını değerlendirmek için doğrulama seti. Sağlanmazsa, `trainset`'in bir kısmı ayrılır (`valset_ratio` tarafından kontrol edilir). Hem `valset` hem de `valset_ratio` None/0 ise, doğrulama yapılmaz ve en son program döndürülür. | `None` |
| `num_threads` | `int \| None` | Paralel değerlendirme iş parçacığı (thread) sayısı. Varsayılan None'dır, bu da sıralı (sequential) değerlendirme anlamına gelir. | `None` |
| `max_errors` | `int \| None` | Değerlendirme sırasında tolere edilecek maksimum hata sayısı. Varsayılanı `dspy.settings.max_errors`'dur. | `None` |
| `provide_traceback` | `bool \| None` | Değerlendirme hataları için detaylı izleme (traceback) gösterilip gösterilmeyeceği. | `None` |
| `seed` | `int \| None` | Tekrarlanabilirlik için rastgele tohum. `trainset` karıştırma (shuffling) ve değerlendirme örneklemesini (sampling) kontrol eder. | `None` |
| `valset_ratio` | `float` | Doğrulama olarak ayrılacak `trainset` oranı ([0, 1) aralığında). Örneğin, 0.1 %10'u ayırır. Doğrulamayı atlamak için 0'a ayarlayın. Varsayılan 0.1'dir. | `0.1` |
| `shuffle_trainset_between_steps` | `bool` | Her optimizasyon adımından önce `trainset`'in karıştırılıp karıştırılmayacağı. Örnek sıralamasına aşırı uyumu (overfitting) önlemeye yardımcı olur. Varsayılan True'dur. | `True` |
| `strategy` | `str` | `" -> "` ile ayrılmış olarak uygulanacak optimize edicilerin sırası. Her eleman `__init__` içinde sağlanan optimize edicilerden bir anahtar (key) olmalıdır. Örneğin, `"p -> w -> p"` önce istem optimizasyonunu, sonra ağırlık optimizasyonunu, ardından tekrar istem optimizasyonunu uygular. Varsayılan `"p -> w -> p"`dir. | `'p -> w -> p'` |
| `optimizer_compile_args` | `dict[str, dict[str, Any]] \| None` | Optimize edici anahtarlarını (keys) kendi `compile()` argümanlarına eşleyen isteğe bağlı sözlük. Belirli bir optimize edici için sözlükte `trainset`, `valset` veya `teacher` sağlanırsa, bunlar BetterTogether'ın compile metodundaki varsayılanları geçersiz kılar (override). Örneğin: `{"p": {"num_trials": 10}, "w": {"trainset": custom_trainset}}`. Bu, belirli optimize ediciler için varsayılan derleme argümanlarını geçersiz kılmak için kullanışlıdır. `student` argümanı `optimizer_compile_args` içine dahil edilemez; BetterTogether'ın compile metodu tüm optimize ediciler için öğrenci referansını yönetir. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Module` | Optimize edilmiş öğrenci programı; iki ek öznitelikle birlikte döner:<br>- **`candidate_programs`**: 'program', 'score' ve 'strategy' anahtarlarına sahip, skora göre sıralanmış (en iyi ilk sırada) sözlük listesi. Temel (baseline) de dahil olmak üzere değerlendirilen tüm programları içerir.<br>- **`flag_compilation_error_occurred`**: Herhangi bir optimizasyon adımının başarısız olup olmadığını gösteren boolean değer. |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | `trainset` boşsa, `valset_ratio` [0, 1) aralığında değilse, `strategy` boşsa veya geçersiz optimize edici anahtarları içeriyorsa veya `optimizer_compile_args` geçersiz argümanlar içeriyorsa fırlatılır. |
| `TypeError` | `optimizer_compile_args` bir 'student' anahtarı içeriyorsa (buna izin verilmez) fırlatılır. |

**Örnekler (Examples):**

```python
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric),
...     w=BootstrapFinetune(metric=metric)
... )
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w"
... )
>>> print(f"Best score: {compiled.candidate_programs[0]['score']}")
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bettertogether.py`*

### `get_params`

```python
get_params() -> dict[str, Any]
```

Teleprompter'ın parametrelerini alır.
**Dönüş Değerleri:** Teleprompter'ın parametreleri.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/teleprompt.py`*

---

## BetterTogether Nasıl Çalışır? (How BetterTogether Works)

BetterTogether, optimize edicileri yapılandırılabilir bir sırayla yürütür, her bir ara sonucu değerlendirir ve en iyi performans gösteren programı döndürür. İşte nasıl çalıştığı:

### 1. Özel Optimize Edicilerle Başlatma (Initialization with Custom Optimizers)
Başlatıldığında (initialized), BetterTogether herhangi bir DSPy optimize edicisini (Teleprompter) anahtar kelime argümanı olarak kabul eder. Anahtarlar, strateji string'inde kullanılan optimize edici adları haline gelir:

```python
optimizer = BetterTogether(
    metric=metric,
    p=GEPA(...),           # stratejide 'p' kullanılabilir
    w=BootstrapFinetune(...) # stratejide 'w' kullanılabilir
)
```

Hiçbir optimize edici sağlanmazsa, BetterTogether varsayılan olarak `BootstrapFewShotWithRandomSearch` (anahtar: `'p'`) ve `BootstrapFinetune` (anahtar: `'w'`) kullanır.

### 2. Strateji Yürütme (Strategy Execution)
Strateji string'i uygulanacak optimize edicilerin sırasını tanımlar. Örneğin:

```python
compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="p -> w -> p"
)
```

Bu strateji (`"p -> w -> p"`) şu anlama gelir:
1. İstem optimize ediciyi çalıştır ('p')
2. Sonuç üzerinde ağırlık optimize ediciyi çalıştır ('w')
3. Sonuç üzerinde istem optimize ediciyi tekrar çalıştır ('p')

Her adımda:
* `trainset` karıştırılır (eğer `shuffle_trainset_between_steps=True` ise)
* Optimize edici, mevcut öğrenci programı üzerinde çalıştırılır
* Sonuç, doğrulama (validation) seti üzerinde değerlendirilir
* Aday program ve skoru kaydedilir

BetterTogether bir meta-optimize edici olduğu için herhangi bir optimize edici dizisi birleştirilebilir. Strateji string'indeki optimize edici adları, başlatma sırasındaki anahtar kelime argümanlarına karşılık gelir. Örneğin, farklı istem optimize edicileri art arda sıralayabilirsiniz (not: bu BetterTogether'ın esnekliğini gösterir, illa ki önerilen bir yapılandırma değildir):

```python
optimizer = BetterTogether(
    metric=metric,
    mipro=MIPROv2(metric=metric, auto="light"),
    gepa=GEPA(metric=metric, auto="light")
)

compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="mipro -> gepa -> mipro"
)
```

### 3. Doğrulama ve Program Seçimi (Validation and Program Selection)
BetterTogether bir doğrulama setini (validation set) üç şekilde kullanabilir:
* **Açık (Explicit) valset:** `valset` sağlanmışsa, değerlendirme için o kullanılır.
* **Otomatik bölme (Auto-split):** `valset_ratio > 0` ise, `trainset`'in bir kısmı doğrulama için ayrılır (held out).
* **Doğrulamasız (No validation):** Hem `valset` hem de `valset_ratio` None/0 ise, doğrulama yapılmaz.

Tüm optimizasyon adımları tamamlandıktan sonra, doğrulama setinin kullanılabilirliğine göre en iyi program seçilir:
* **Doğrulama ile:** **En iyi skora** sahip program döndürülür (eşitlik durumunda daha önceki programlar kazanır).
* **Doğrulama olmadan:** **En son (latest) program** döndürülür.

Eğer bir optimizasyon adımı başarısız olursa:
* Hata, tam izlemeyle (full traceback) günlüğe kaydedilir.
* Optimizasyon erken durur.
* O ana kadar bulunan en iyi program döndürülür.
* `flag_compilation_error_occurred` değeri `True` olarak ayarlanır.

Döndürülen program iki ek öznitelik içerir:
* **`candidate_programs`**: Skora göre sıralanmış (en iyi ilk sırada), skorları ve stratejileri ile birlikte değerlendirilen tüm programların listesi. Bir hata oluştuğunda, bu, arıza noktasına kadar başarıyla değerlendirilen tüm programları içerir.
* **`flag_compilation_error_occurred`**: Derleme (compilation) sırasında herhangi bir optimizasyon adımının başarısız olup olmadığını gösteren boolean değer.

### 4. Daha Fazla Detay (Further Details)
**Model Yaşam Döngüsü Yönetimi:** Yerel modeller için (LocalLM gibi), BetterTogether modelleri ilk kullanımdan önce otomatik olarak başlatır (launches), optimizasyon tamamlandıktan sonra öldürür (kills) ve model adları değiştiğinde ince ayardan sonra yeniden başlatır (relaunches). Bu işlemler API tabanlı LLM'ler için etkisizdir ancak yerel model sunumu (local model serving) için gereklidir.

**Özel Derleme Argümanları (Custom Compile Arguments):** `optimizer_compile_args` parametresini kullanarak belirli optimize edicilere özel derleme argümanları aktarabilirsiniz:
* **Varsayılan argümanları geçersiz kıl (Override):** Belirli optimize edicilere özel trainset/valset/teacher aktarın.
* **Optimize edici başına özelleştir:** Her optimize edici farklı derleme argümanlarına (örn. `num_trials`, `max_bootstrapped_demos`) sahip olabilir.

**Not:** `student` argümanı `optimizer_compile_args` içine dahil edilemez; BetterTogether, tüm optimize ediciler için öğrenci programını yönetir. Detaylı argüman belgelemesi için `compile()` metodu docstring'ine bakın.

---

## En İyi Uygulamalar (Best Practices)

### BetterTogether Ne Zaman Kullanılmalı
BetterTogether şu durumlarda doğru optimize edicidir:
* **Performansın her zerresini sıkmak istediğinizde:** İstem optimizasyonu genellikle paranın karşılığını en iyi veren yöntemdir ve yüksek seviyeli stratejileri hızla keşfeder. Fırsat elverdiğinde, bunun üzerine ağırlık optimizasyonunu (ince ayar) eklemek bu kazanımları katlar ve her iki yaklaşımın da tek başına yapabileceğinden daha fazla fayda sağlar.
* **İnce ayar (fine-tuning) yetenekleriniz olduğunda:** `BootstrapFinetune` gibi ağırlık optimize ediciler, bir ince ayar arayüzüne sahip LLM'ler gerektirir. Şu anda desteklenenler: `LocalProvider`, `DatabricksProvider` ve `OpenAIProvider`. Özel kullanım durumları için `Provider` sınıfını genişletebilir veya BetterTogether'ı yalnızca istem optimize edicilerini birleştirmek için kullanabilirsiniz.

[Databricks vaka çalışması] bu etkililiği ortaya koymaktadır. Kurumsal alanları (finans, hukuk, ticaret, sağlık hizmetleri) kapsayan ve karmaşık zorluklar (100+ sayfalık belgeler, 70+ çıkarma alanı ve hiyerarşik şemalar) barındıran kapsamlı bir paket olan IE Bench üzerinde değerlendirme yaptılar. GPT-4.1 kullanarak:
* **Sadece SFT (Supervised Fine-Tuning):** Temel modele (baseline) göre +1.9 puan
* **Sadece GEPA:** Temel modele göre +2.1 puan (SFT'yi biraz geçiyor)
* **GEPA + SFT (BetterTogether):** Temel modele göre +4.8 puan

Bu, istem optimizasyonunun denetimli ince ayarla eşleşebileceğini veya onu aşabileceğini ve bu tekniklerin birleştirilmesinin güçlü bileşik faydalar (compounding benefits) sağladığını göstermektedir.

### Yaygın Stratejiler ve Optimize Ediciler (Common Strategies and Optimizers)
Yaygın stratejiler:
* `"p -> w"`: Önce istemleri optimize edin, sonra ince ayar yapın (basit ve genellikle etkilidir)
* `"p -> w -> p"`: İstemleri optimize edin, ince ayar yapın, ardından istemleri tekrar optimize edin (ince ayar iyileştirmelerinin üzerine inşa edilebilir)
* `"w -> p"`: Önce ince ayar yapın, sonra istemleri optimize edin

Örnek optimize edici kombinasyonları:
* **GEPA + BootstrapFinetune:** İnce ayar ile birleştirilmiş istem optimizasyonu
* **MIPROv2 + BootstrapFinetune:** İnce ayar ile birleştirilmiş istem optimizasyonu
* **Çoklu istem optimize ediciler:** Farklı istem optimizasyon yaklaşımları arasında geçiş yapın (deneysel)

## Daha Fazla Okuma (Further Reading)
* **BetterTogether Makalesi:** arxiv:2407.10930  https://arxiv.org/abs/2407.10930
* **Databricks Case Study:** BetterTogether ile GEPA'yı birleştiren gerçek dünya uygulaması  https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization
* **DSPy Optimizers Overview**  https://dspy.ai/learn/programming/optimizers.md
