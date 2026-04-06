# Eğitim: Ajanlarda Fine-tuning

50 adımlı görevlerle bir oyun oynayan ReAct ajanını temsil eden bir DSPy modülü içindeki _dil modeli ağırlıklarını_ (yani fine-tuning’i) optimize etmeye yönelik hızlı bir örneği adım adım inceleyelim.

### Bağımlılıkları yükleme ve veriyi indirme

En güncel DSPy sürümünü `pip install -U dspy` ile kurun ve takip edin. Bu eğitim, DSPy 2.6.0’a bağlı olan AlfWorld veri kümesini kullanır.

Ayrıca aşağıdaki bağımlılıklara da ihtiyacınız olacak:

```shell
> pip install -U alfworld==0.3.5 multiprocess
> alfworld-download
```

<details>
<summary>Önerilir: Arka planda neler olduğunu öğrenmek için MLflow Tracing kurun</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, DSPy’nin davranışını daha iyi anlamak için MLflow’u kullanarak istemleri ve optimizasyon ilerlemesini izler halinde görselleştirebilirsiniz. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz.

![MLflow Trace](./mlflow-tracing-agent.png)

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow UI’ı başlatın
```bash
mlflow ui --port 5000
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```

Entegrasyon hakkında daha fazla bilgi edinmek için ayrıca [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını ziyaret edin.
</details>

### Dil modellerini ayarlama

Amacımız, `gpt-4o-mini` modelinin AlfWorld ev içi oyununu, dize istemlerini veya örnek izlekleri elle kurcalamadan yetkin biçimde oynayabilmesini sağlamak.

Kesin olarak gerekli olmasa da, daha büyük olan `gpt-4o` modelini istem optimizasyonu ve fine-tuning için kullanarak, küçük `gpt-4o-mini` ajanımızı oluşturarak işimizi biraz kolaylaştıracağız.

```python
import dspy

gpt4o_mini = dspy.LM('gpt-4o-mini-2024-07-18')
gpt4o = dspy.LM('openai/gpt-4o')
dspy.configure(experimental=True)
```

AlfWorld’den 200 eğitim ve 200 geliştirme görevi yükleyelim. Veri kümesi çok daha büyük, ancak az sayıda örnek kullanmak bu eğitimin, fine-tuning dahil, 1–2 saat içinde tamamlanmasına yardımcı olacaktır.

Yalnızca 100 eğitim göreviyle, 4o-mini’nin başarısını %19’dan (oyunu zar zor oynayabiliyor) %72’ye çıkaracağız. 500 görev kullanır ve fine-tuning sırasında gösterimleri korursanız bunu kolayca %82’ye çıkarabilirsiniz.

```python
from dspy.datasets.alfworld import AlfWorld

alfworld = AlfWorld()
trainset, devset = alfworld.trainset[:200], alfworld.devset[-200:]
len(trainset), len(devset)
```

Devam etmeden önce bu görevin bir örneğine bakalım.

```python
example = trainset[0]

with alfworld.POOL.session() as env:
    task, info = env.init(**example.inputs())

print(task)
```

### Ajan programını tanımlama

Ajan, `self.react` adlı tek bir alt modüle sahip oldukça basit bir `dspy.Module`’dür.

Bu alt modül, belirli bir `task` tanımını alır, önceki `trajectory` bilgisini görür ve gerçekleştirebileceği `possible_actions` listesini görür. Basitçe bir sonraki eylemle yanıt verir.

`forward` metodunda ise verilen `idx` görevi için bir ortam başlatırız. Ardından `self.max_iters` sayısına kadar döngüye girer, her seferinde bir sonraki eylemi almak için `self.react` modülünü tekrar tekrar çağırırız.

```python
class Agent(dspy.Module):
    def __init__(self, max_iters=50, verbose=False):
        self.max_iters = max_iters
        self.verbose = verbose
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

    def forward(self, idx):
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            if self.verbose:
                print(f"Task: {task}")

            for _ in range(self.max_iters):
                trajectory_ = "\n".join(trajectory)
                possible_actions = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                prediction = self.react(task=task, trajectory=trajectory_, possible_actions=possible_actions)
                trajectory.append(f"> {prediction.action}")

                if prediction.action.startswith("think:"):
                    trajectory.append("OK.")
                    continue

                obs, reward, done, info = env.step(prediction.action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)

                if self.verbose:
                    print("\n".join(trajectory[-2:]))

                if done:
                    break

        assert reward == int(info["won"][0]), (reward, info["won"][0])
        return dspy.Prediction(trajectory=trajectory, success=reward)
```

#### Not: Ajanınıza talimat eklemek isteseydiniz...

Yukarıda ajanı, görevi tarif eden kısa talimatlar bile vermeden son derece basit tuttuk.

Prensipte, AlfWorld görevinin kısa bir tanımını (Yao ve diğerleri, 2022 temel alınarak) kopyalayıp bunu ajanınız için talimat olarak kullanabilirsiniz. Bu doğası gereği zorunlu değildir, fakat talimatların DSPy’deki rolünü göstermeye yardımcı olur: modelin belirli bir davranışı sergilemesini zorlamak için değil, görevin temellerini yalın ve insan tarafından okunabilir bir biçimde açıklamak için vardır.

Bunu yapmak isterseniz, şu satırı:

```python
self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")
```

şununla değiştirebilirsiniz:

```python
INSTRUCTIONS = """
Yüksek seviyeli bir hedefe ulaşmak için simüle edilmiş bir ev ortamıyla etkileşime geç. Plan yaptığından, alt hedefleri takip ettiğinden,
yaygın ev eşyalarının olası konumlarını belirlediğinden (ör. masa lambaları muhtemelen masalarda, raflarda veya çekmeceliklerde olur)
ve sistematik biçimde keşif yaptığından emin ol (ör. masa lambası için tüm masaları tek tek kontrol et).
""".strip()

self.react = dspy.Predict(dspy.Signature("task, trajectory, possible_actions: list[str] -> action", INSTRUCTIONS))
```

### Zero-shot değerlendirme

Şimdi, herhangi bir optimizasyon yapmadan önce bu basit programı deneyelim.

```python
agent_4o = Agent()
agent_4o.set_lm(gpt4o)
agent_4o.verbose = True

agent_4o(**example.inputs())
```

Tamam, bu durumda bu örneği çözemedi! Şimdi 4o ve 4o-mini’nin ortalama kalitesine bakalım.

```python
metric = lambda x, y, trace=None: y.success
evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, num_threads=16)
```

<details>
<summary>MLflow Experiment içinde değerlendirme sonuçlarını izleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.

```python
import mlflow

with mlflow.start_run(run_name="agent_evaluation"):
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=16,
        display_progress=True,
        # Çıktıları ve ayrıntılı skorları MLflow’a kaydetmek için
        return_all_scores=True,
        return_outputs=True,
    )

    # Programı her zamanki gibi değerlendir
    aggregated_score, outputs, all_scores = evaluate(cot)

    # Toplu skoru kaydet
    mlflow.log_metric("success_rate", aggregated_score)
    # Ayrıntılı değerlendirme sonuçlarını tablo olarak kaydet
    mlflow.log_table(
        {
            "Idx": [example.idx for example in eval_set],
            "Result": outputs,
            "Success": all_scores,
        },
        artifact_file="eval_results.json",
    )
```

Entegrasyon hakkında daha fazla bilgi edinmek için ayrıca [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını ziyaret edin.

</details>

```python
agent_4o.verbose = False
evaluate(agent_4o)
```

```python
agent_4o_mini = Agent()
agent_4o_mini.set_lm(gpt4o_mini)

evaluate(agent_4o_mini)
```

Kutudan çıktığı haliyle bu görevde 4o iyi sayılır (%58 başarı oranı), ancak 4o-mini zorlanır (%15 başarı oranı).

Şu stratejiyi uygulayalım:

1. Önce _promptları_ hafif bir şekilde gpt-4o için optimize edeceğiz.
2. Ardından bu prompt-optimize edilmiş ajanı öğretmen olarak kullanıp görev üzerinde gpt-4o-mini’ye fine-tuning uygulayacağız. Bu, kalitesini %19’dan %72’ye çıkaracak (ya da 500 trainset örneği kullanırsanız %82’ye).

### GPT-4o için prompt optimizasyonu

```python
optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=16, prompt_model=gpt4o)

config = dict(max_bootstrapped_demos=1, max_labeled_demos=0, minibatch_size=40)
optimized_4o = optimizer.compile(agent_4o, trainset=trainset, **config)
```

### GPT-4o-mini’ye fine-tuning uygulama

Fine-tuning için bir öğretmen programa (`optimized_4o`) ve ondan türetilmiş bir öğrenci programa (aşağıdaki `student_4om`) ihtiyacımız olacak.

```python
student_4o_mini = optimized_4o.deepcopy()
student_4o_mini.set_lm(gpt4o_mini)
# student_4o_mini.react.demos = []  # isterseniz gösterimleri sıfırlayabilirsiniz
```

```python
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=16)
finetuned_4o_mini = optimizer.compile(student_4o_mini, teacher=optimized_4o, trainset=trainset)
```

### Fine-tuning uygulanmış GPT-4o-mini ajanını değerlendirme

```python
evaluate(finetuned_4o_mini)
```

Tüm bu optimizasyonu yaptıktan sonra, programımızı daha sonra kullanabilmek için kaydedelim. Sağlayıcı tarafında aynı tanımlayıcıyla varlığını sürdürdüğü sürece, bu işlem fine-tune edilmiş modele de bir referans tutacaktır.

```python
finetuned_4o_mini.save('finetuned_4o_mini_001.pkl')
```

<details>
<summary>Programları MLflow Experiment içinde kaydetme</summary>

<br/>

Programı yerel bir dosyaya kaydetmek yerine, daha iyi yeniden üretilebilirlik ve iş birliği için onu MLflow’da takip edebilirsiniz.

1. **Bağımlılık Yönetimi**: MLflow, yeniden üretilebilirliği sağlamak için dondurulmuş ortam meta verisini programla birlikte otomatik olarak kaydeder.
2. **Deney Takibi**: MLflow ile programın performansını ve maliyetini, programın kendisiyle birlikte takip edebilirsiniz.
3. **İş Birliği**: MLflow deneyini paylaşarak programı ve sonuçları ekip üyelerinizle paylaşabilirsiniz.

Programı MLflow içine kaydetmek için aşağıdaki kodu çalıştırın:

```python
import mlflow

# Bir MLflow Run başlat ve programı kaydet
with mlflow.start_run(run_name="optimized"):
    model_info = mlflow.dspy.log_model(
        finetuned_4o_mini,
        artifact_path="model", # Programı MLflow içinde kaydetmek için herhangi bir ad
    )

# Programı MLflow’dan tekrar yükle
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

Entegrasyon hakkında daha fazla bilgi edinmek için ayrıca [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını ziyaret edin.

</details>

Şimdi fine-tune edilmiş ajan programımızla bir göreve bakalım!

```python
finetuned_4o_mini.verbose = True
finetuned_4o_mini(**devset[0].inputs())
```

Ajan programını yükleyip kullanmak isterseniz, bunu aşağıdaki gibi yapabilirsiniz.

> **⚠️ Güvenlik Uyarısı:** `.pkl` dosyalarını yüklemek rastgele kod çalıştırabilir ve tehlikeli olabilir. Pickle dosyalarını yalnızca güvenilir kaynaklardan ve güvenli ortamlarda kaydedip yükleyin. Daha güvenli serileştirme için mümkün olduğunda JSON biçimini kullanmayı değerlendirin.

```python
loaded = Agent()
loaded.load('finetuned_4o_mini_001.pkl', allow_pickle=True)
```
