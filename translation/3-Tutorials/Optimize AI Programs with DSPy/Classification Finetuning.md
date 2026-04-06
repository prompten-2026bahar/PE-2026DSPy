# Eğitim: Sınıflandırma Fine-tuning

Bir DSPy programı içindeki LM ağırlıklarını fine-tune etmeye yönelik hızlı bir örneği adım adım inceleyelim. Bunu basit bir 77 sınıflı sınıflandırma görevine uygulayacağız.

Fine-tune edilmiş programımız, GPU’nuzda yerel olarak barındırılan küçük bir `Llama-3.2-1B` dil modelini kullanacak. Bunu daha ilginç hale getirmek için, (i) elimizde **hiç eğitim etiketi olmadığını** ama (ii) etiketlenmemiş 500 eğitim örneğimiz olduğunu varsayacağız.

### Bağımlılıkları yükleme ve veriyi indirme

En güncel DSPy sürümünü `pip install -U dspy` ile kurun ve birlikte ilerleyin (isterseniz `uv pip` de kullanabilirsiniz). Bu eğitim DSPy >= 2.6.0 sürümüne bağlıdır. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekir.

Bu eğitim şu anda çıkarım için yerel bir GPU gerektiriyor; ancak future’da fine-tune edilmiş modeller için ollama serving desteği de eklemeyi planlıyoruz.

Ayrıca aşağıdaki bağımlılıklara ihtiyacınız olacak:
1. Çıkarım: Yerel çıkarım sunucularını çalıştırmak için SGLang kullanıyoruz. En güncel sürümü kurmak için buradaki talimatları izleyebilirsiniz: https://docs.sglang.ai/start/install.html  
Aşağıda 04/02/2025 itibarıyla en güncel kurulum komutu paylaşılmıştır; ancak kurulum bağlantısına giderek en güncel sürümdeki talimatları izlemenizi öneririz.  
Bu, fine-tuning paketleri ile `sglang` paketinin birbiriyle uyumlu olmasını sağlar.
    ```shell
    > pip install --upgrade pip
    > pip install uv
    > uv pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
    ```
2. Fine-tuning: Aşağıdaki paketleri kullanıyoruz. `transformers` paketi için sürüm belirttiğimizi unutmayın; bu, yakın zamanda ortaya çıkan bir sorun için geçici çözümdür: https://github.com/huggingface/trl/issues/2338
    ```shell
    > uv pip install -U torch transformers==4.48.3 accelerate trl peft
    ```

Kurulumu hızlandırmak için `uv` paket yöneticisini kullanmanızı öneririz.

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. MLflow’u aşağıdaki dört adımı izleyerek kolayca kurabilirsiniz.

![MLflow Trace](./mlflow-tracing-classification.png)

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow arayüzünü başlatın
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

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

### Veri kümesi

Bu eğitim için Banking77 veri kümesini kullanacağız.

```python
import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset

# Banking77 veri kümesini yükle.
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

# Veri kümesinden ilk 2000 örneği yükle ve her *eğitim* örneğine bir ipucu ata.
raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:1000]
]

random.Random(0).shuffle(raw_data)
```

Bu veri kümesinde sınıflandırma için 77 farklı kategori vardır. Bunlardan bazılarını gözden geçirelim.

```python
len(CLASSES), CLASSES[:10]
```

Banking77 içinden 500 adet (etiketlenmemiş) sorgu örnekleyelim. Bunları bootstrap edilmiş fine-tuning için kullanacağız.

```python
unlabeled_trainset = [dspy.Example(text=x.text).with_inputs("text") for x in raw_data[:500]]

unlabeled_trainset[0]
```

### DSPy programı

Diyelim ki `text` alanını alıp adım adım akıl yürüten ve ardından Banking77 sınıflarından birini seçen bir program istiyoruz.

Bunun esas olarak gösterim amacıyla ya da modelin akıl yürütmesini incelemek istediğiniz, örneğin az da olsa açıklanabilirlik istediğiniz durumlar için tasarlandığını unutmayın. Başka bir deyişle, bu tür görevlerin açık akıl yürütmeden mutlaka çok büyük fayda sağlayacağı garanti değildir.

```python
from typing import Literal

classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")
```

### Bootstrap edilmiş fine-tuning

Bunu yapmanın birçok yolu vardır; örneğin modelin kendine öğretmesine izin vermek ya da etiketler olmadan yüksek güvenli durumları belirlemek için çıkarım-zamanı hesaplama kullanmak (örneğin ensemble yöntemleri).

Belki de en basit yol, bu görevde makul iş çıkaracağını düşündüğümüz bir modeli akıl yürütme ve sınıflandırma için öğretmen olarak kullanmak ve bunu küçük modelimize damıtmaktır. Bu kalıpların tümü birkaç satır kodla ifade edilebilir.

Küçük `Llama-3.2-1B-Instruct` modelini öğrenci LM olarak ayarlayalım. Öğretmen LM olarak GPT-4o-mini kullanacağız.

```python
from dspy.clients.lm_local import LocalProvider

student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
student_lm = dspy.LM(model=f"openai/local:{student_lm_name}", provider=LocalProvider(), max_tokens=2000)
teacher_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=3000)
```

Şimdi bu LM’lerimize sınıflandırıcıları atayalım.

```python
student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

teacher_classify = classify.deepcopy()
teacher_classify.set_lm(teacher_lm)
```

Şimdi bootstrap edilmiş fine-tuning’i başlatalım. Buradaki “bootstrap edilmiş” ifadesi, programın eğitim girdileri üzerinde çalıştırılacağı ve tüm modüller üzerindeki ortaya çıkan izlerin kaydedilip fine-tuning için kullanılacağı anlamına gelir. Bu, DSPy’deki çeşitli BootstrapFewShot yöntemlerinin ağırlık optimize eden varyantıdır.

Etiketlenmemiş eğitim kümesindeki her soru için bu, öğretmen programı çalıştıracak; öğretmen program da akıl yürütme üretecek ve bir sınıf seçecektir. Bu süreç izlenecek ve ardından öğrenci programındaki tüm modüller için (bu örnekte yalnızca tek bir CoT modülü) bir eğitim kümesi oluşturacaktır.

`compile` metodu çağrıldığında, `BootstrapFinetune` optimizer’ı, iletilen öğretmen programı (veya programları; bir liste de verebilirsiniz!) kullanarak bir eğitim veri kümesi oluşturacaktır.  
Ardından bu eğitim veri kümesini, `student` programı için ayarlanan LM’nin fine-tune edilmiş bir sürümünü oluşturmak üzere kullanacak ve bunu eğitilmiş LM ile değiştirecektir.  
Eğitilmiş LM’nin yeni bir LM örneği olacağını unutmayın (burada oluşturduğumuz `student_lm` nesnesi dokunulmadan kalacaktır!)

Not: Elinizde etiketler varsa, `BootstrapFinetune` yapıcısına `metric` geçebilirsiniz. Bunu pratikte uygulamak istiyorsanız, yerel LM eğitim ayarlarını kontrol etmek için yapıcıya `train_kwargs` geçebilirsiniz: `device`, `use_peft`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `max_seq_length`, `packing`, `bf16` ve `output_dir`.

```python
# İsteğe bağlı:
# [1] Checkpoint’lerin ve fine-tuning verisinin saklanacağı dizini kontrol etmek için
#     `DSPY_FINETUNEDIR` ortam değişkenini ayarlayabilirsiniz.
#     Bu ayarlanmazsa varsayılan olarak `DSPY_CACHEDIR` kullanılır.
# [2] Fine-tuning ve çıkarım için kullanılacak GPU’yu kontrol etmek amacıyla
#     `CUDA_VISIBLE_DEVICES` ortam değişkenini ayarlayabilirsiniz.
#     Bu ayarlanmazsa ve HuggingFace `transformers` kütüphanesinin kullandığı varsayılan GPU doluysa,
#     OutOfMemoryError oluşabilir.
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["DSPY_FINETUNEDIR"] = "/path/to/dir"
```

```python
dspy.settings.experimental = True  # fine-tuning deneysel bir özelliktir, bu yüzden etkinleştirmek için bir bayrak ayarlıyoruz

optimizer = dspy.BootstrapFinetune(num_threads=16)  # eğer *etiketleriniz varsa*, buraya metric=your_metric geçin!
classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=unlabeled_trainset)
```

Bu yerel bir model olduğu için, onu açıkça başlatmamız gerekir.

```python
classify_ft.get_lm().launch()
```

### Fine-tune edilmiş programı doğrulama

Şimdi bunun başarılı olup olmadığını anlamaya çalışalım. Sisteme bir soru sorabilir ve davranışını inceleyebiliriz.

```python
classify_ft(text="Paramı daha önce almadım ve işlem hâlâ devam ediyor görünüyor. Bunu düzeltebilir misiniz?")
```

Küçük bir altın etiket kümesi de alabilir ve sistemin görülmemiş sorgulara genelleme yapıp yapamadığına bakabiliriz.

```python
devset = raw_data[500:600]
devset[0]
```

Bu küçük geliştirme kümesi üzerinde bir değerlendirici tanımlayalım; burada metrik, akıl yürütmeyi yok sayacak ve etiketin tam olarak doğru olup olmadığını kontrol edecektir.

```python
metric = (lambda x, y, trace=None: x.label == y.label)
evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, display_table=5, num_threads=16)
```

Şimdi fine-tune edilmiş 1B sınıflandırıcıyı değerlendirelim.

```python
evaluate(classify_ft)
```

<details>
<summary>MLflow Experiment içinde değerlendirme sonuçlarını izleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.

```python
import mlflow

with mlflow.start_run(run_name="classifier_evaluation"):
    evaluate_correctness = dspy.Evaluate(
        devset=devset,
        metric=extraction_correctness_metric,
        num_threads=16,
        display_progress=True,
    )

    # Programı her zamanki gibi değerlendir
    result = evaluate_correctness(people_extractor)

    # Toplu skoru kaydet
    mlflow.log_metric("exact_match", result.score)
    # Ayrıntılı değerlendirme sonuçlarını tablo olarak kaydet
    mlflow.log_table(
        {
            "Text": [example.text for example in devset],
            "Expected": [example.example_label for example in devset],
            "Predicted": [output[1] for output in result.results],
            "Exact match": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

Görev için hiç etiketle başlamamış olmamıza rağmen fena değil. Etiketiniz olmasa bile, bootstrap edilmiş eğitim verisinin kalitesini artırmak için çeşitli stratejiler kullanabilirsiniz.

Bunu sonraki adımda denemek için, fine-tune edilmiş LM’yi kapatarak GPU belleğimizi boşaltalım.

```python
classify_ft.get_lm().kill()
```

### Bir metrik karşısında bootstrap edilmiş fine-tuning

Etiketleriniz varsa, genel olarak bunu büyük ölçüde iyileştirebilirsiniz. Bunu yapmak için `BootstrapFinetune` içine bir `metric` geçebilirsiniz; optimizer bu metriği, fine-tuning verisini oluşturmadan önce programınız üzerindeki trajectory’leri filtrelemek için kullanacaktır.

```python
optimizer = dspy.BootstrapFinetune(num_threads=16, metric=metric)
classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=raw_data[:500])
```

Şimdi bunu başlatalım ve değerlendirelim.

```python
classify_ft.get_lm().launch()
```

```python
evaluate(classify_ft)
```

Yalnızca 500 etiketle bile bu oldukça daha iyi. Hatta öğretmen LM’nin kutudan çıktığı halinden çok daha güçlü görünüyor!

```python
evaluate(teacher_classify)
```

Bootstrap sayesinde model, modüllerimizi kullanarak doğru etiketi nasıl elde edeceğini öğreniyor; bu örnekte bunu açık akıl yürütme ile yapıyor:

```python
classify_ft(text="kartım neden hâlâ gelmedi?")
dspy.inspect_history()
```

<details>
<summary>Fine-tune edilmiş programları MLflow Experiment içine kaydetme</summary>

<br/>

Fine-tune edilmiş programı üretimde dağıtmak veya ekibinizle paylaşmak için onu MLflow Experiment içine kaydedebilirsiniz. Yalnızca yerel bir dosyaya kaydetmeye kıyasla MLflow şu avantajları sunar:

1. **Bağımlılık Yönetimi**: MLflow, yeniden üretilebilirliği sağlamak için dondurulmuş ortam meta verisini programla birlikte otomatik olarak kaydeder.
2. **Deney Takibi**: MLflow ile programın performansını ve maliyetini, programın kendisiyle birlikte takip edebilirsiniz.
3. **İş Birliği**: MLflow deneyini paylaşarak programı ve sonuçları ekip üyelerinizle paylaşabilirsiniz.

Programı MLflow içine kaydetmek için aşağıdaki kodu çalıştırın:

```python
import mlflow

# Bir MLflow Run başlat ve programı kaydet
with mlflow.start_run(run_name="optimized_classifier"):
    model_info = mlflow.dspy.log_model(
        classify_ft,
        artifact_path="model", # Programı MLflow içinde kaydetmek için herhangi bir ad
    )

# Programı MLflow’dan tekrar yükle
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>
