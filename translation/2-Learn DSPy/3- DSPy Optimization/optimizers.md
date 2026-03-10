# DSPy Optimizer'ları (Eski adıyla Teleprompter'lar)

Bir **DSPy optimizer** (iyileştirici), doğruluk gibi belirttiğiniz metrikleri maksimize etmek için bir DSPy programının parametrelerini (yani istemleri ve/veya LM ağırlıklarını) ayarlayabilen bir algoritmadır.

Tipik bir DSPy optimizer üç şey alır:

- **DSPy programınız**. Bu, tek bir modül (örneğin `dspy.Predict`) veya karmaşık, çok modüllü bir program olabilir.

- **Metriğiniz**. Bu, programınızın çıktısını değerlendiren ve ona bir puan (yüksek olması daha iyidir) atayan bir fonksiyondur.

- Birkaç **eğitim girdisi**. Bu çok küçük olabilir (yani sadece 5 veya 10 örnek) ve eksik olabilir (yalnızca programınıza gelen girdiler, herhangi bir etiket olmadan).

Çok fazla veriniz varsa, DSPy bundan yararlanabilir. Ancak küçük başlayıp güçlü sonuçlar elde edebilirsiniz.

**Not:** Eskiden teleprompter olarak adlandırılıyordu. Kütüphane ve dökümantasyon genelinde yansıtılacak resmi bir isim güncellemesi yapıyoruz.

## Bir DSPy Optimizer Neyi Ayarlar? Onları Nasıl Ayarlar?


DSPy'deki farklı optimizer'lar, her modül için `dspy.BootstrapRS`<sup>[1]</sup> gibi **iyi az-örnekli (few-shot) örnekler sentezleyerek**, `dspy.MIPROv2`<sup>[2]</sup> ve `dspy.GEPA`<sup>[3]</sup> gibi her istem (prompt) için **daha iyi doğal dil talimatları önerip akıllıca keşfederek** ve `dspy.BootstrapFinetune`<sup>[4]</sup> gibi sisteminizdeki **modüller için veri setleri oluşturup bunları LM ağırlıklarına ince ayar (finetune) yapmak için kullanarak** programınızın kalitesini ayarlar.

??? "Bir DSPy optimizer örneği nedir? Farklı optimizer'lar nasıl çalışır?"

    Örnek olarak `dspy.MIPROv2` optimizer'ını ele alalım. İlk olarak, MIPRO **özyükleme (bootstrapping) aşaması** ile başlar. Bu noktada optimize edilmemiş olabilecek programınızı alır ve her bir modülünüz için girdi/çıktı davranışı izlerini (traces) toplamak üzere farklı girdiler üzerinde defalarca çalıştırır. Bu izleri, metriğiniz tarafından yüksek puan alan yörüngelerde görünenleri tutacak şekilde filtreler. İkinci olarak, MIPRO **temellendirilmiş öneri (grounded proposal) aşamasına** girer. DSPy programınızın kodunu, verilerinizi ve programınızı çalıştırmadan elde edilen izleri önizler ve bunları programınızdaki her istem için birçok potansiyel talimat taslağı hazırlamak için kullanır. Üçüncü olarak, MIPRO **ayrık arama (discrete search) aşamasını** başlatır. Eğitim setinizden mini paketler (mini-batches) örnekler, boru hattındaki her istemi oluşturmak için kullanılacak talimat ve iz kombinasyonlarını önerir ve aday programı mini paket üzerinde değerlendirir. Elde edilen puanı kullanarak, MIPRO önerilerin zamanla daha iyi hale gelmesine yardımcı olan bir vekil modeli (surrogate model) günceller.


    DSPy optimizer'larını bu kadar güçlü kılan şeylerden biri de birleştirilebilir olmalarıdır. `dspy.MIPROv2`'yi çalıştırabilir ve üretilen programı tekrar `dspy.MIPROv2`'ye veya örneğin daha iyi sonuçlar almak için `dspy.BootstrapFinetune`'a girdi olarak kullanabilirsiniz. Bu, kısmen `dspy.BetterTogether` yaklaşımının özüdür. Alternatif olarak, optimizer'ı çalıştırıp ardından en iyi 5 aday programı çıkarabilir ve bunlardan bir `dspy.Ensemble` (topluluk) oluşturabilirsiniz. Bu, *çıkarım zamanı hesaplamasını* (örneğin topluluklar) ve DSPy'nin benzersiz *çıkarım öncesi hesaplamasını* (yani optimizasyon bütçesi) oldukça sistematik yollarla ölçeklendirmenize olanak tanır.



## Şu Anda Hangi DSPy Optimizer'ları Mevcut?

Optimizer'lara `from dspy.teleprompt import *` üzerinden erişilebilir.

### Otomatik Az-Örnekli Öğrenme (Automatic Few-Shot Learning)

Bu optimizer'lar, az-örnekli öğrenmeyi uygulayarak modele gönderilen istem (prompt) içine otomatik olarak **optimize edilmiş** örnekler oluşturup dahil ederek imzayı (signature) genişletir.


1. [**`LabeledFewShot`**](../../api/optimizers/LabeledFewShot.md): Sağlanan etiketli girdi ve çıktı veri noktalarından basitçe az-örnekli örnekler (demolar) oluşturur. Rastgele `k` adet örnek seçmek için `k` (istem için örnek sayısı) ve `trainset` gerektirir.

2. [**`BootstrapFewShot`**](../../api/optimizers/BootstrapFewShot.md): Programınızın her aşaması için tam gösterimler oluşturmak üzere bir `teacher` (öğretmen) modülü (varsayılan olarak kendi programınızdır) ve `trainset` içindeki etiketli örnekleri kullanır. Parametreler arasında `max_labeled_demos` (`trainset` içinden rastgele seçilen gösterim sayısı) ve `max_bootstrapped_demos` (`teacher` tarafından oluşturulan ek örnek sayısı) yer alır. Özyükleme (bootstrapping) süreci, gösterimleri doğrulamak için metriği kullanır ve yalnızca metriği geçenleri "derlenmiş" isteme dahil eder. İleri seviye: Daha zor görevler için uyumlu yapıya sahip *farklı* bir DSPy programı olan bir `teacher` programı kullanmayı destekler.

3. [**`BootstrapFewShotWithRandomSearch`**](../../api/optimizers/BootstrapFewShotWithRandomSearch.md): Oluşturulan gösterimler üzerinde rastgele arama yaparak `BootstrapFewShot` işlemini birkaç kez uygular ve optimizasyon süresince en iyi programı seçer. Parametreler `BootstrapFewShot` ile aynıdır; ek olarak optimizasyon sırasında değerlendirilen rastgele program sayısını belirten `num_candidate_programs` parametresine sahiptir.

4. [**`KNNFewShot`**](../../api/optimizers/KNNFewShot.md): Verilen bir girdi örneği için en yakın eğitim örneği gösterimlerini bulmak amacıyla k-En Yakın Komşu (k-Nearest Neighbors) algoritmasını kullanır. Bu en yakın komşu gösterimleri daha sonra `BootstrapFewShot` optimizasyon süreci için eğitim seti (trainset) olarak kullanılır.


### Otomatik Talimat Optimizasyonu (Automatic Instruction Optimization)

Bu optimizer'lar, istem (prompt) için en uygun talimatları üretir ve MIPROv2 durumunda az-örnekli (few-shot) gösterim setini de optimize edebilir.



5. [**`COPRO`**](../../api/optimizers/COPRO.md): Her adım için yeni talimatlar üretip bunları iyileştirir ve koordinat yükselmesi (metrik fonksiyonu ve `trainset` kullanarak tepe tırmanma - hill-climbing) ile optimize eder. Parametreler, optimizer'ın üzerinden geçtiği istem iyileştirme iterasyonlarının sayısı olan `depth` değerini içerir.

6. [**`MIPROv2`**](../../api/optimizers/MIPROv2.md): Her adımda talimatlar *ve* az-örnekli örnekler üretir. Talimat üretimi veri ve gösterim duyarlıdır (data-aware & demonstration-aware). Modülleriniz genelindeki üretim talimatları/gösterimleri alanında etkili bir şekilde arama yapmak için Bayesyen Optimizasyonu (Bayesian Optimization) kullanır.


7. [**`SIMBA`**](../../api/optimizers/SIMBA.md): Yüksek çıktı değişkenliğine sahip zorlu örnekleri belirlemek için stokastik mini paket örneklemesi (stochastic mini-batch sampling) kullanır, ardından hataları içsel olarak analiz etmek ve öz-yansıtmalı iyileştirme kuralları oluşturmak veya başarılı gösterimler eklemek için Dil Modelini (LLM) uygular.

8. [**`GEPA`**](../../api/optimizers/GEPA/overview.md): Nelerin işe yaradığını, nelerin yaramadığını belirlemek ve boşlukları gideren istemler önermek için DSPy programının yörüngesi üzerinde düşünmek üzere Dil Modellerini kullanır. Ek olarak GEPA, DSPy programını hızla geliştirmek için alana özgü metinsel geri bildirimlerden yararlanabilir. GEPA kullanımıyla ilgili ayrıntılı eğitimlere [dspy.GEPA Tutorials](../../tutorials/gepa_ai_program/index.md) adresinden ulaşılabilir.

### Otomatik İnce Ayar (Automatic Finetuning)

Bu optimizer, altta yatan Dil Modelini (LLM) ince ayar (fine-tune) yapmak için kullanılır.



9. [**`BootstrapFinetune`**](/api/optimizers/BootstrapFinetune): İstem tabanlı (prompt-based) bir DSPy programını ağırlık güncellemelerine damıtır (distill). Çıktı, aynı adımlara sahip olan ancak her adımın istemli bir LM yerine ince ayar yapılmış bir model tarafından yürütüldüğü bir DSPy programıdır. Tam bir örnek için [sınıflandırma ince ayar eğitimine](https://dspy.ai/tutorials/classification_finetuning/) bakabilirsiniz.

### Program Dönüşümleri (Program Transformations)

10. [**`Ensemble`**](../../api/optimizers/Ensemble.md): Bir dizi DSPy programını topluluk haline getirir (ensemble) ve ya tam seti kullanır ya da bir alt kümeyi rastgele örnekleyerek tek bir programda birleştirir.

### Meta-Optimizer'lar

11. [**`BetterTogether`**](../../api/optimizers/BetterTogether.md): İstem optimizasyonu ve ağırlık optimizasyonunu (ince ayar) yapılandırılabilir diziler halinde birleştiren bir meta-optimizer'dır. İstem optimizasyonu etkili görev ayrıştırma ve akıl yürütme stratejilerini keşfedebilirken; ağırlık optimizasyonu, modeli bu kalıpları daha verimli yürütmesi için özelleştirebilir. Bu yaklaşımları diziler halinde (örneğin; istem → ağırlık → istem) birlikte kullanmak, her birinin diğerinin yaptığı iyileştirmeler üzerine inşa edilmesini sağlayabilir. Deneysel olarak bu yaklaşım, genellikle her iki stratejinin tek başına kullanımından daha iyi performans gösterir. Ayrıntılı eğitime [BetterTogether AIME Tutorial](../../tutorials/bettertogether_aime/index.ipynb) adresinden ulaşılabilir.

## Hangi optimizer'ı kullanmalıyım?



Nihayetinde, göreviniz için "doğru" optimizer'ı ve en iyi yapılandırmayı bulmak deneme gerektirecektir. DSPy'de başarı hala yinelemeli bir süreçtir; görevinizde en iyi performansı elde etmek, keşfetmenizi ve yineleme yapmanızı gerektirecektir.

Bununla birlikte, başlamak için genel rehberlik şöyledir:

- Eğer **çok az örneğiniz** (yaklaşık 10 tane) varsa, `BootstrapFewShot` ile başlayın.
- Eğer **daha fazla veriniz** (50 örnek veya daha fazla) varsa, `BootstrapFewShotWithRandomSearch` aracını deneyin.
- Eğer **sadece talimat optimizasyonu** yapmak istiyorsanız (yani isteminizi 0-shot tutmak istiyorsanız), [0-shot optimizasyonu için yapılandırılmış](../../api/optimizers/MIPROv2.md) `MIPROv2` kullanın.
- Eğer **daha uzun optimizasyon çalışmaları** (örneğin 40 deneme veya daha fazla) yürütmek için daha fazla çıkarım çağrısı kullanmaya istekliyseniz ve yeterli veriniz varsa (örneğin aşırı uyumu önlemek için 200 örnek veya daha fazla), `MIPROv2`'yi deneyin.
- Bunlardan birini büyük bir LM (örneğin 7B parametreli veya üzeri) ile kullanabildiyseniz ve çok **verimli bir programa** ihtiyacınız varsa, `BootstrapFinetune` ile göreviniz için küçük bir LM'ye ince ayar yapın.

## Bir optimizer'ı nasıl kullanırım?

Hepsi, anahtar kelime argümanlarında (hiperparametreler) bazı farklılıklar olsa da bu genel arayüzü paylaşır. Tam listeye [API referansından](../../api/optimizers/BetterTogether.md) ulaşılabilir.

Bunu en yaygın olanı, `BootstrapFewShotWithRandomSearch` ile görelim.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Optimizer'ı kurun: Programınızın adımları için 8-örnekli (8-shot) örnekleri "özyüklemek" (yani kendi kendine oluşturmak) istiyoruz.
# Optimizer, geliştirme setindeki (devset) en iyi denemesini seçmeden önce bunu 10 kez (artı bazı başlangıç denemeleri) tekrarlayacaktır.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=SIZIN_METRIGINIZ, **config)
optimized_program = teleprompter.compile(SIZIN_PROGRAMINIZ, trainset=SIZIN_EGITIM_SETINIZ)
```


!!! info "Başlangıç III: DSPy Programlarında LM İstemlerini veya Ağırlıklarını Optimize Etme"
    Tipik bir basit optimizasyon çalışması yaklaşık **2 USD** maliyetindedir ve yaklaşık on dakika sürer; ancak optimizer'ları çok büyük LM'ler veya çok büyük veri setleri ile çalıştırırken dikkatli olun.
    Optimizer çalışmaları, LM'nize, veri setinize ve yapılandırmanıza bağlı olarak birkaç sent kadar az veya onlarca dolar kadar yüksek maliyetli olabilir.
    
    === "ReAct ajanı için istemleri optimize etme"
        Bu, Wikipedia üzerinden arama yaparak soruları yanıtlayan bir `dspy.ReAct` ajanı kurmaya ve ardından onu `HotPotQA` veri setinden örneklenen 500 soru-cevap çifti üzerinde ekonomik `light` modunda `dspy.MIPROv2` kullanarak optimize etmeye yönelik minimal ancak tamamen çalıştırılabilir bir örnektir.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        DSPy 2.5.29 üzerinde yapılan benzer bir kayıt dışı çalışma, ReAct ajanının başarısını %24'ten %51'e çıkarmaktadır.

    === "RAG için istemleri optimize etme"
    Bir arama dizini (`search`), favori Dil Modeliniz (`dspy.LM`) ve sorulardan oluşan küçük bir eğitim setiniz (`trainset`) olduğunda, aşağıdaki kod parçacığı uzun çıktılı RAG sisteminizi yerleşik `dspy.SemanticF1` metriğine göre optimize edebilir. Bu metrik, bir DSPy modülü olarak uygulanmıştır ve yanıtın hem doğruluğunu hem de kapsamını semantik olarak değerlendirir.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # not defined in this snippet, see link above
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        Çalıştırabileceğiniz eksiksiz bir RAG örneği için bu [eğiticiye (tutorial)](../../tutorials/rag/index.ipynb) başlayabilirsiniz. Bu örnek, StackExchange topluluklarının bir alt kümesi üzerindeki RAG sisteminin kalitesini %53'ten %61'e çıkarmaktadır.

    === "Sınıflandırma için ağırlıkları optimize etme (Finetuning)"
    <details><summary>Veri seti kurulum kodunu göstermek için tıklayın.</summary>

        ```python linenums="1"
        import random
        from typing import Literal

        from datasets import load_dataset

        import dspy
        from dspy.datasets import DataLoader

        # 1. Banking77 veri setini yükle ve etiket isimlerini (CLASSES) al.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
        kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

        # 2. Veri setinden ilk 2000 örneği yükle.
# Her bir *eğitim* örneğine bir 'hint' (ipucu) atayarak Example nesnelerini oluştur.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        ```python linenums="1"
        import dspy
        lm=dspy.LM('openai/gpt-4o-mini-2024-07-18')

        # Define the DSPy module for classification. It will use the hint at training time, if available.
        signature = dspy.Signature("text, hint -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThought(signature)
        classify.set_lm(lm)

        # Optimize via BootstrapFinetune.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="What does a pending cash withdrawal mean?")
        
        # For a complete fine-tuning tutorial, see: https://dspy.ai/tutorials/classification_finetuning/
        ```

        **Olası Çıktı:**
        ```text
        Prediction(
            reasoning='A pending cash withdrawal indicates that a request to withdraw cash has been initiated but has not yet been completed or processed. This status means that the transaction is still in progress and the funds have not yet been deducted from the account or made available to the user.',
            label='pending_cash_withdrawal'
        )
        ```

        DSPy 2.5.29 üzerindeki buna benzer kayıt dışı bir çalışma, GPT-4o-mini'nin puanını %66'dan %87'ye çıkarmaktadır.


## Optimizer çıktısını kaydetme ve yükleme

Bir programı bir optimizer üzerinden çalıştırdıktan sonra, onu kaydetmek de faydalıdır. Daha sonraki bir noktada, bir program bir dosyadan yüklenebilir ve çıkarım (inference) için kullanılabilir. Bunun için `load` ve `save` yöntemleri kullanılabilir.

```python
optimized_program.save(YOUR_SAVE_PATH)
```

Sonuç dosyası düz metin JSON formatındadır. Kaynak programdaki tüm parametreleri ve adımları içerir. İstediğiniz zaman okuyabilir ve iyileştiricinin (optimizer) ne oluşturduğunu görebilirsiniz.

Bir programı bir dosyadan yüklemek için, ilgili sınıftan bir nesne oluşturabilir ve ardından bu nesne üzerinde load (yükle) yöntemini çağırabilirsiniz.

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```

