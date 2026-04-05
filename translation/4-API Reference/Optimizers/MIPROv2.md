# /translation/mipro_v2.md

## dspy.MIPROv2

**MIPROv2** (Multiprompt Instruction PRoposal Optimizer Version 2 / Çoklu-İstem Talimat Önerisi Optimize Edici Versiyon 2), hem talimatları hem de az atışlı (few-shot) örnekleri ortaklaşa optimize edebilen bir istem (prompt) optimize edicisidir. Bunu, az atışlı örnek adaylarını önyükleyerek (bootstrapping), görevin farklı dinamiklerine dayanan talimatlar önererek ve Bayes Optimizasyonu (Bayesian Optimization) kullanarak bu seçeneklerin optimize edilmiş bir kombinasyonunu bularak yapar. Hem az atışlı örnekleri ve talimatları ortaklaşa optimize etmek için hem de yalnızca 0-atışlı (0-shot) optimizasyon için talimatları optimize etmek amacıyla kullanılabilir.

```python
dspy.MIPROv2(metric: Callable, prompt_model: Any | None = None, task_model: Any | None = None, teacher_settings: dict | None = None, max_bootstrapped_demos: int = 4, max_labeled_demos: int = 4, auto: Literal['light', 'medium', 'heavy'] | None = 'light', num_candidates: int | None = None, num_threads: int | None = None, max_errors: int | None = None, seed: int = 9, init_temperature: float = 1.0, verbose: bool = False, track_stats: bool = True, log_dir: str | None = None, metric_threshold: float | None = None)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/mipro_optimizer_v2.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student: Any, *, trainset: list, teacher: Any = None, valset: list | None = None, num_trials: int | None = None, max_bootstrapped_demos: int | None = None, max_labeled_demos: int | None = None, seed: int | None = None, minibatch: bool = True, minibatch_size: int = 35, minibatch_full_eval_steps: int = 5, program_aware_proposer: bool = True, data_aware_proposer: bool = True, view_data_batch_size: int = 10, tip_aware_proposer: bool = True, fewshot_aware_proposer: bool = True, requires_permission_to_run: bool | None = None, provide_traceback: bool | None = None) -> Any
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/mipro_optimizer_v2.py`*

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

Aşağıdaki program, bir matematik programının MIPROv2 ile nasıl optimize edileceğini göstermektedir:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Optimize ediciyi içe aktarın
from dspy.teleprompt import MIPROv2

# LM'i başlatın
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# Optimize ediciyi başlatın
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # light, medium ve heavy optimizasyon çalıştırmaları arasından seçim yapılabilir
)

# Programı optimize edin
print(f"Program MIPROv2 ile optimize ediliyor...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# Optimize edilmiş programı gelecekte kullanmak üzere kaydedin
optimized_program.save(f"optimized.json")
```

---

## MIPROv2 Nasıl Çalışır? (How MIPROv2 works)

Üst düzeyde bakıldığında `MIPROv2`, LM programınızdaki her bir tahminci (predictor) için hem az atışlı (few-shot) örnekler hem de yeni talimatlar oluşturarak ve ardından programınız için bu değişkenlerin en iyi kombinasyonunu bulmak üzere Bayes Optimizasyonu (Bayesian Optimization) kullanarak bunlar üzerinde arama yaparak çalışır. Eğer görsel bir açıklama isterseniz bu twitter zincirine (thread) göz atabilirsiniz.



Bu adımlar aşağıda daha ayrıntılı olarak açıklanmıştır:

**1) Önyüklemeli Az Atışlı Örnekler (Bootstrap Few-Shot Examples):** Eğitim setinizden rastgele örnekler alır ve bunları LM programınızdan geçirir. Programdan çıkan çıktı bu örnek için doğruysa, geçerli bir az atışlı örnek adayı olarak tutulur. Aksi takdirde, belirtilen miktarda az atışlı örnek adayı derleyene kadar başka bir örnek deneriz. Bu adım, `max_bootstrapped_demos` önyüklenmiş örnekten ve eğitim setinden örneklenen `max_labeled_demos` temel örnekten oluşan `num_candidates` kadar set oluşturur.

**2) Talimat Adayları Önerme (Propose Instruction Candidates):** Talimat önericisi; (1) eğitim veri setinin özelliklerinin oluşturulmuş bir özetini, (2) LM programınızın kodunun ve bir talimatın kendisi için üretildiği spesifik tahmincinin oluşturulmuş bir özetini, (3) belirli bir tahminci için referans girdileri / çıktıları göstermek üzere daha önce önyüklenmiş az atışlı örnekleri ve (4) potansiyel talimatların özellik alanını keşfetmeye yardımcı olmak için rastgele örneklenmiş bir üretme ipucunu (örneğin "yaratıcı ol", "kısa ol" vb.) içerir. Bu bağlam, yüksek kaliteli talimat adayları yazan bir `prompt_model`'e sağlanır.

**3) Az Atışlı Örneklerin ve Talimatların Optimize Edilmiş Bir Kombinasyonunu Bulma (Find an Optimized Combination of Few-Shot Examples & Instructions):** Son olarak, programımızdaki her bir tahminci için hangi talimat ve gösterim (demonstrations) kombinasyonlarının en iyi işe yaradığını seçmek için Bayes Optimizasyonunu kullanırız. Bu, bir dizi `num_trials` deneme çalıştırılarak yapılır, burada her denemede yeni bir istem (prompts) seti doğrulama setimiz (validation set) üzerinden değerlendirilir. Yeni istem seti her denemede yalnızca `minibatch_size` büyüklüğündeki bir mini yığın (minibatch) üzerinde değerlendirilir (`minibatch=True` olduğunda). Ortalamada en iyi performansı gösteren istem seti, ardından her `minibatch_full_eval_steps` adımında bir tam doğrulama seti üzerinde değerlendirilir. Optimizasyon sürecinin sonunda, tam doğrulama setinde en iyi performansı gösteren istem setine sahip LM programı döndürülür.

Daha fazla detayla ilgilenenler için, `MIPROv2` hakkındaki daha fazla bilgi ve `MIPROv2`'nin diğer DSPy optimize edicileriyle karşılaştırıldığı bir çalışma bu makalede bulunabilir.
