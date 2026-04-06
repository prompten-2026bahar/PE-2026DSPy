# Eğitim: Çok Adımlı Araştırma için Çevrim İçi RL

UYARI: Bu özellik yeni ve son derece DENEYSELDİR. DSPy’deki neredeyse her şeyin aksine, şu anda tamamen kavram kanıtı ve geliştirme modundadır; ancak topluluk katılımını teşvik etmek için bunu yayımlıyoruz.

Bu eğitim için ayrıca [DSPy’nin Arbor RL framework’üne](https://github.com/Ziems/arbor) de ihtiyacınız olacak; bunu şu komutla kurabilirsiniz:

```bash
> pip install -U arbor-ai
```

DSPy’yi ana daldan kurmanız da gerekebilir:
```bash
> pip install -U git+https://github.com/stanfordnlp/dspy.git@main
```

```python
import dspy
import arbor
from arbor import ArborGRPO, ArborProvider
arbor_server_info = arbor.init() # Arbor sunucusunu arka planda başlat

port = 7453
local_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_lm = dspy.LM(
    model=f"openai/arbor:{local_lm_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
    # Arbor bunların eğitim yapılandırmasıyla eşleştiğini kontrol eder
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=2048,
)

dspy.configure(lm=local_lm)
```

### Bağımlılıkları yükleme ve veriyi indirme

Bilgi getirme işlemi için hafif bir kütüphane olduğu için harika BM25S kütüphanesini kullanacağız. İsterseniz bu bileşenleri istediğiniz başka şeylerle değiştirebilirsiniz.

```shell
> pip install -U bm25s PyStemmer "jax[cpu]"
```

Ardından, 2017 itibarıyla tüm 5.000.000 Wikipedia sayfasının özetlerinden (yani ilk paragraflarından) oluşan bir anlık görüntüyü indireceğiz. Bunu getirme külliyatımız olarak kullanacağız.

Bu dosya sıkıştırılmış hâlde 500MB boyutundadır; bu yüzden indirme ve açma işlemi 2-3 dakika sürebilir.

```python
from dspy.utils import download

download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
!tar -xzvf wiki.abstracts.2017.tar.gz
```

Ardından bunu BM25 getirme için indeksleyelim! Bu da 2-3 dakika sürecektir.

```python
import orjson
import bm25s
import Stemmer

corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = orjson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)
```

### HoVer veri kümesini yükleme

Görevimiz için bir veri kümesi yükleyelim. HoVer çok adımlı görevinden örnekler yükleyeceğiz; burada girdi (gerçekten de!) karmaşık bir iddia, aradığımız çıktı ise bu iddiayı doğrulamak için gereken Wikipedia sayfalarının kümesidir.

Bunu düzgün çalıştırmak için veri kümesinin daha eski bir sürümünü yüklemeniz gerekebilir...
```shell
> pip install datasets==3.6.0
```

```python
import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:600], hover[600:900], hover[900:]
len(trainset), len(devset), len(testset)
```

Şimdi Wikipedia içinde arama yapmak için bir fonksiyon tanımlayalım. Bu, BM25 indeksimizi kullanacaktır.

```python
def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return list(run.keys())
```

## Çok adımlı araştırma için bir DSPy programı

Şimdi DSPy içinde çok adımlı araştırma programını tanımlayalım. Son derece basit olacak; `generate_query` ve `append_notes` modüllerinden oluşacak. Talimatları dikkatlice tanımlayacağız, ancak bunlar genellikle zorunlu değildir.

```python
instr1 = """
Bir iddia ve bazı temel olgular verildiğinde, iddiayı doğrulama veya çürütme yönünde bir sonraki en gerekli ipucunu bulmak için bir takip arama sorgusu üret. Nihai amaç, iddiayla ilişkili tüm belgeleri bulmaktır.
""".strip()

instr2 = """
Bir iddia, bazı temel olgular ve yeni arama sonuçları verildiğinde, yeni arama sonuçlarından elde edilen yeni öğrenimleri belirle. Bunlar, iddianın doğru mu yanlış mı olduğuna dair şimdiye kadar bilinen temel olguları genişletecektir. Nihai amaç, iddiayla ilişkili tüm belgeleri bulmamıza yardımcı olacak tüm olguları toplamaktır.
"""


class ResearchHop(dspy.Module):
    def __init__(self, num_docs, num_hops):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(dspy.Signature("claim, key_facts -> followup_search_query", instr1))
        self.append_notes = dspy.ChainOfThought(dspy.Signature("claim, key_facts, new_search_results -> new_key_facts", instr2))

    def forward(self, claim: str) -> list[str]:
        key_facts = []
        retrieved_docs = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(claim=claim, key_facts=key_facts).followup_search_query if hop_idx else claim
            search_results = search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)

            if hop_idx == self.num_hops - 1:
                break

            prediction = self.append_notes(claim=claim, key_facts=key_facts, new_search_results=search_results)
            key_facts.append(prediction.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)
```

### Bu görevde başarı için metrikleri tanımlama

```python
def recall(example, pred, trace=None):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)

evaluate = dspy.Evaluate(devset=devset, metric=recall, num_threads=16, display_progress=True, display_table=5)
```

## `ResearchHop` sistemini `dspy.GRPO` ile optimize etme

```python
program = ResearchHop(num_docs=4, num_hops=2)
program.set_lm(local_lm)

# NOT: Eğitim 4 GPU üzerinde yapılıyor.
train_kwargs = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 24/6,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "beta": 0.00,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "loss_type": "dapo",
    "max_steps": 1000,
    "report_to": "wandb",
    "log_completions": True,
    "logging_steps": 1,
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": False,
    "max_grad_norm": 1.0,
    "lora_config": {
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    },
    "num_training_gpus": 3,
    "num_inference_gpus": 1,
    "weight_decay": 0.001,
}

compiler = ArborGRPO(
    metric=recall,
    num_dspy_examples_per_grpo_step=6,
    num_rollouts_per_grpo_step=24,
    exclude_demos=True,
    num_train_steps=1000,
    num_threads=16,
    use_train_as_val=False,
    num_steps_for_val=50,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

optimized_program = compiler.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)
```

Artık GRPO uygulanmış programı kullanabilirsiniz.

```python
example = devset[0]
optimized_program(**example.inputs())
```

İlk deneylerimizde, yaklaşık 18 saatlik eğitim recall değerini (devset üzerinde) %61,8’den %66,2’ye yükseltiyor. Bu, maliyet/kalite açısından _genellikle_ `dspy.MIPROv2` veya `dspy.SIMBA` gibi istem optimize edicileri çalıştırmaktan elde edeceğiniz sonuçlardan daha zayıftır; ancak küçük LM’ler için rastgele LM programları üzerinde çevrim içi RL adına yine de çok sağlam bir başlangıçtır.
