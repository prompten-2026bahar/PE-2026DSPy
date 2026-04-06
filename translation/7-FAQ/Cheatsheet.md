# DSPy Hızlı Başvuru

Bu sayfa, sık kullanılan örüntüler için kod parçaları içerecektir.

## DSPy Programları

### Yeni LM Çıktılarını Zorlamak

DSPy, LM çağrılarını önbelleğe alır. Mevcut bir önbellek girdisini aşmak ve yine de yeni sonucu önbelleğe almak için benzersiz bir `rollout_id` sağlayın ve sıfırdan farklı bir `temperature` değeri (ör. `1.0`) ayarlayın:

```python
predict = dspy.Predict("question -> answer")
predict(question="1+1", config={"rollout_id": 1, "temperature": 1.0})
```

### dspy.Signature

```python
class BasicQA(dspy.Signature):
    """Soruları kısa, bilgi odaklı yanıtlarla cevapla."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="çoğunlukla 1 ile 5 kelime arasında")
```

### dspy.ChainOfThought

```python
generate_answer = dspy.ChainOfThought(BasicQA)

# Belirli bir girdide kestiriciyi çağır.
question='Gökyüzünün rengi nedir?'
pred = generate_answer(question=question)
```

### dspy.ProgramOfThought

```python
pot = dspy.ProgramOfThought(BasicQA)

question = 'Sarah’nin 5 elması var. Markete gidip 7 elma daha alıyor. Sarah’nin şimdi kaç elması var?'
result = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
```

### dspy.ReAct

```python
react_module = dspy.ReAct(BasicQA)

question = 'Sarah’nin 5 elması var. Markete gidip 7 elma daha alıyor. Sarah’nin şimdi kaç elması var?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```

### dspy.Retrieve

```python
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(rm=colbertv2_wiki17_abstracts)

# Retrieve modülünü tanımla
retriever = dspy.Retrieve(k=3)

query='İlk FIFA Dünya Kupası ne zaman düzenlendi?'

# Belirli bir sorgu üzerinde retriever'ı çağır.
topK_passages = retriever(query).passages

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

### dspy.CodeAct

```python
from dspy import CodeAct

def factorial(n):
    """n'in faktöriyelini hesapla"""
    if n == 1:
        return 1
    return n * factorial(n-1)

act = CodeAct("n->factorial", tools=[factorial])
result = act(n=5)
result # 120 döndürür
```

### dspy.Parallel

```python
import dspy

parallel = dspy.Parallel(num_threads=2)
predict = dspy.Predict("question -> answer")
result = parallel(
    [
        (predict, dspy.Example(question="1+1").with_inputs("question")),
        (predict, dspy.Example(question="2+2").with_inputs("question"))
    ]
)
result
```

## DSPy Metrikleri

### Metrik Olarak Fonksiyon

Özel bir metrik oluşturmak için sayı veya boolean döndüren bir fonksiyon tanımlayabilirsiniz:

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]

        # içinde sayı olan son token'ı bul
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer

# Metrik Fonksiyonu
def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### Hakem Olarak LLM

```python
class FactJudge(dspy.Signature):
    """Yanıtın bağlama göre olgusal olarak doğru olup olmadığını değerlendir."""

    context = dspy.InputField(desc="Tahmin için bağlam")
    question = dspy.InputField(desc="Yanıtlanacak soru")
    answer = dspy.InputField(desc="Soruya verilen yanıt")
    factually_correct: bool = dspy.OutputField(desc="Yanıt bağlama göre olgusal olarak doğru mu?")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return factual.factually_correct
```

## DSPy Değerlendirme

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```

## DSPy Optimize Edicileri

### LabeledFewShot

```python
from dspy.teleprompt import LabeledFewShot

labeled_fewshot_optimizer = LabeledFewShot(k=8)
your_dspy_program_compiled = labeled_fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

### BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Derleme için başka bir LM kullanma, teacher_settings ile belirtme

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10, teacher_settings=dict(lm=gpt4))

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Derlenmiş bir programı derleme - bootstrap edilmiş bir programı yeniden bootstrap etme

```python
your_dspy_program_compiledx2 = teleprompter.compile(
    your_dspy_program,
    teacher=your_dspy_program_compiled,
    trainset=trainset,
)
```

#### Derlenmiş bir programı kaydetme/yükleme

```python
save_path = './v1.json'
your_dspy_program_compiledx2.save(save_path)
```

```python
loaded_program = YourProgramClass()
loaded_program.load(path=save_path)
```

### BootstrapFewShotWithRandomSearch

BootstrapFewShotWithRandomSearch hakkında ayrıntılı dokümantasyon [burada](api/optimizers/BootstrapFewShot.md) bulunabilir.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

```

Diğer özel yapılandırmalar, `BootstrapFewShot` optimize edicisini özelleştirmeye benzer.

### Ensemble

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)
your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
programs = [x[-1] for x in your_dspy_program_compiled.candidate_programs]
your_dspy_program_compiled_ensemble = ensemble_optimizer.compile(programs[:3])
```

### BootstrapFinetune

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

# Geçerli dspy.settings.lm üzerinde programı derle
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
your_dspy_program_compiled = tp.compile(your_dspy_program, trainset=trainset[:some_num], valset=trainset[some_num:])

# Fine-tune edilecek modeli yapılandır
config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)

# Programı BootstrapFinetune ile derle
finetune_optimizer = BootstrapFinetune(metric=your_defined_metric)
finetune_program = finetune_optimizer.compile(your_dspy_program, trainset=some_new_dataset_for_finetuning_model, **config)

finetune_program = your_dspy_program

# Programı yükle ve değerlendirme öncesi programdaki model parametrelerini etkinleştir
ckpt_path = "saved_checkpoint_path_from_finetuning"
LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)

for p in finetune_program.predictors():
    p.lm = LM
    p.activated = False
```

### COPRO

COPRO hakkında ayrıntılı dokümantasyon [burada](api/optimizers/COPRO.md) bulunabilir.

```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

copro_teleprompter = COPRO(prompt_model=model_to_generate_prompts, metric=your_defined_metric, breadth=num_new_prompts_generated, depth=times_to_generate_prompts, init_temperature=prompt_generation_temperature, verbose=False)

compiled_program_optimized_signature = copro_teleprompter.compile(your_dspy_program, trainset=trainset, eval_kwargs=eval_kwargs)
```

### MIPROv2

Not: ayrıntılı dokümantasyon [burada](api/optimizers/MIPROv2.md) bulunabilir. `MIPROv2`, `MIPRO`’nun en güncel uzantısıdır ve (1) talimat önerisinde iyileştirmeler ve (2) mini-batch ile daha verimli arama gibi güncellemeler içerir.

#### MIPROv2 ile optimizasyon

Bu örnek, birçok hiperparametreyi sizin için ayarlayan ve hafif bir optimizasyon çalıştırması gerçekleştiren `auto=light` ile kolay bir başlangıç kullanımını gösterir. Alternatif olarak daha uzun optimizasyon çalışmaları için `auto=medium` veya `auto=heavy` ayarlayabilirsiniz. [Buradaki](api/optimizers/MIPROv2.md) daha ayrıntılı MIPROv2 dokümantasyonu, hiperparametrelerin elle nasıl ayarlanacağına dair daha fazla bilgi de sağlar.

```python
# Optimize ediciyi içe aktar
from dspy.teleprompt import MIPROv2

# Optimize ediciyi başlat
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # light, medium ve heavy optimizasyon çalışmaları arasında seçim yapılabilir
)

# Programı optimize et
print(f"Program MIPRO ile optimize ediliyor...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
)

# Gelecekte kullanmak için optimize edilmiş programı kaydet
optimized_program.save(f"mipro_optimized")

# Optimize edilmiş programı değerlendir
print(f"Optimize edilmiş program değerlendiriliyor...")
evaluate(optimized_program, devset=devset[:])
```

#### MIPROv2 ile yalnızca talimatları optimize etme (0-Shot)

```python
# Optimize ediciyi içe aktar
from dspy.teleprompt import MIPROv2

# Optimize ediciyi başlat
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # light, medium ve heavy optimizasyon çalışmaları arasında seçim yapılabilir
)

# Programı optimize et
print(f"Program MIPRO ile optimize ediliyor...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

# Gelecekte kullanmak için optimize edilmiş programı kaydet
optimized_program.save(f"mipro_optimized")

# Optimize edilmiş programı değerlendir
print(f"Optimize edilmiş program değerlendiriliyor...")
evaluate(optimized_program, devset=devset[:])
```

### KNNFewShot

```python
from sentence_transformers import SentenceTransformer
from dspy import Embedder
from dspy.teleprompt import KNNFewShot
from dspy import ChainOfThought

knn_optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))

qa_compiled = knn_optimizer.compile(student=ChainOfThought("question -> answer"))
```

### BootstrapFewShotWithOptuna

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

fewshot_optuna_optimizer = BootstrapFewShotWithOptuna(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optuna_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```

Diğer özel yapılandırmalar, `dspy.BootstrapFewShot` optimize edicisini özelleştirmeye benzer.

### SIMBA

SIMBA, Stochastic Introspective Mini-Batch Ascent anlamına gelir. Bu, keyfi DSPy programlarını kabul eden ve istem talimatlarında veya few-shot örneklerde artımlı iyileştirmeler aramak için mini-batch dizileri halinde ilerleyen bir prompt optimize edicisidir.

```python
from dspy.teleprompt import SIMBA

simba = SIMBA(metric=your_defined_metric, max_steps=12, max_demos=10)

optimized_program = simba.compile(student=your_dspy_program, trainset=trainset)
```

## DSPy Araçları ve Yardımcıları

### dspy.Tool

```python
import dspy

def search_web(query: str) -> str:
    """Bilgi için web'de ara"""
    return f"Arama sonuçları: {query}"

tool = dspy.Tool(search_web)
result = tool(query="Python programlama")
```

### dspy.streamify

```python
import dspy
import asyncio

predict = dspy.Predict("question->answer")

stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```

### dspy.asyncify

```python
import dspy

dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

asyncio.run(dspy_program(question="DSPy nedir"))
```

### Kullanımı İzleme

```python
import dspy
dspy.configure(track_usage=True)

result = dspy.ChainOfThought(BasicQA)(question="2+2 nedir?")
print(f"Token kullanımı: {result.get_lm_usage()}")
```

### dspy.configure_cache

```python
import dspy

# Önbellek ayarlarını yapılandır
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

## DSPy `Refine` ve `BestofN`

>`dspy.Suggest` ve `dspy.Assert`, DSPy 2.6 sürümünde `dspy.Refine` ve `dspy.BestofN` ile değiştirilmiştir.

### BestofN

Bir modülü farklı rollout ID’leriyle (önbelleği aşarak) `N` kereye kadar çalıştırır ve `reward_fn` tarafından tanımlanan en iyi tahmini ya da `threshold` eşiğini geçen ilk tahmini döndürür.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="Belçika'nın başkenti nedir?").answer
# Brussels
```

### Refine

Bir modülü farklı rollout ID’leriyle (önbelleği aşarak) `N` kereye kadar çalıştırarak iyileştirir ve `reward_fn` tarafından tanımlanan en iyi tahmini ya da `threshold` eşiğini geçen ilk tahmini döndürür. Her denemeden sonra (son deneme hariç), `Refine` modülün performansı hakkında otomatik olarak ayrıntılı geri bildirim üretir ve bu geri bildirimi sonraki çalıştırmalar için ipucu olarak kullanır; böylece yinelemeli bir iyileştirme süreci oluşur.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="Belçika'nın başkenti nedir?").answer
# Brussels
```

#### Hata Yönetimi

Varsayılan olarak `Refine`, eşik sağlanana kadar modülü en fazla `N` kez çalıştırmayı dener. Modül hatayla karşılaşırsa, `N` başarısız denemeye kadar devam eder. Bu davranışı `fail_count` değerini `N`’den daha küçük bir sayıya ayarlayarak değiştirebilirsiniz.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=1)
...
refine(question="Belçika'nın başkenti nedir?")
# Yalnızca bir başarısız deneme olursa modül hata fırlatır.
```

Modülü herhangi bir hata yönetimi olmadan en fazla `N` kez çalıştırmak isterseniz, `fail_count` değerini `N` yapabilirsiniz. Bu varsayılan davranıştır.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=3)
...
refine(question="Belçika'nın başkenti nedir?")
```
