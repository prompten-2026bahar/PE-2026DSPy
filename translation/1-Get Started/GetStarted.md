
![DSPy](../img/dspy_logo.png)

## *LM'leri*—istatistiksel komutlarla değil—*Programlamak*

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dspy) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)

DSPy, modüler yapay zeka yazılımları oluşturmak için bildirimsel bir çerçevedir. Kırılgan metin dizileri yerine **yapılandırılmış kod üzerinde hızlıca yineleme yapmanıza** olanak tanır ve ister basit sınıflandırıcılar, ister karmaşık RAG boru hatları veya Ajan döngüleri oluşturuyor olun, **yapay zeka programlarını dil modelleriniz için etkili istemlere ve ağırlıklara derleyen** algoritmalar sunar.

İstemlerle (prompts) uğraşmak veya eğitim işleriyle boğuşmak yerine, DSPy (Bildirimsel Kendi Kendine İyileşen Python), **doğal dil modüllerinden yapay zeka yazılımları oluşturmanıza** ve bunları farklı modeller, çıkarım stratejileri veya öğrenme algoritmalarıyla _genel bir şekilde birleştirmenize_ olanak tanır. Bu, yapay zeka yazılımlarını modeller ve stratejiler arasında **daha güvenilir, sürdürülebilir ve taşınabilir** hale getirir.

*tl;dr* DSPy'yi, assembly'den C'ye veya işaretçi aritmetiğinden SQL'e geçiş gibi, yapay zeka programlama için daha üst düzey bir dil olarak düşünün. [GitHub](https://github.com/stanfordnlp/dspy) ve [Discord](https://discord.gg/XCGy2WDCQB) aracılığıyla toplulukla tanışın, yardım isteyin veya katkıda bulunmaya başlayın.

<!--  Soyutlamaları, yapay zeka yazılımınızı daha güvenilir ve sürdürülebilir hale getirir; ayrıca yeni modeller ve öğrenme teknikleri ortaya çıktıkça yazılımınızın daha taşınabilir olmasına olanak tanır. Üstelik oldukça zariftir! -->

!!! info "Başlarken I: DSPy Kurulumu ve LM Yapılandırması"

    ```bash
    > pip install -U dspy
    ```

    === "OpenAI"
        `OPENAI_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM("openai/gpt-5-mini", api_key="YOUR_OPENAI_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        `ANTHROPIC_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", api_key="YOUR_ANTHROPIC_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        Databricks platformundaysanız, kimlik doğrulaması kendi SDK'ları aracılığıyla otomatik olarak yapılır. Değilseniz, `DATABRICKS_API_KEY` ve `DATABRICKS_API_BASE` ortam değişkenlerini ayarlayabilir veya aşağıdaki gibi `api_key` ve `api_base` parametrelerini iletebilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM(
            "databricks/databricks-llama-4-maverick",
            api_key="YOUR_DATABRICKS_ACCESS_TOKEN",
            api_base="YOUR_DATABRICKS_WORKSPACE_URL",  # örn: [https://dbc-64bf4923-e39e.cloud.databricks.com/serving-endpoints](https://dbc-64bf4923-e39e.cloud.databricks.com/serving-endpoints)
        )
        dspy.configure(lm=lm)
        ```

    === "Gemini"
        `GEMINI_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM("gemini/gemini-2.5-flash", api_key="YOUR_GEMINI_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "Dizüstü bilgisayarınızdaki Yerel LM'ler"
          Öncelikle, [Ollama](https://github.com/ollama/ollama) kurun ve LM'niz ile sunucusunu başlatın.

          ```bash
          > curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh
          > ollama run llama3.2:1b
          ```

          Ardından, DSPy kodunuzdan ona bağlanın.

        ```python linenums="1"
        import dspy
        lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=lm)
        ```

    === "GPU sunucusundaki Yerel LM'ler"
          Öncelikle, [SGLang](https://docs.sglang.ai/get_started/install.html) kurun ve LM'niz ile sunucusunu başlatın.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i [https://flashinfer.ai/whl/cu121/torch2.4/](https://flashinfer.ai/whl/cu121/torch2.4/) 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct
          ```
        
        Meta'dan `meta-llama/Llama-3.1-8B-Instruct` indirmek için erişiminiz yoksa, örnek olarak `Qwen/Qwen2.5-7B-Instruct` kullanabilirsiniz.

        Ardından, yerel LM'nize DSPy kodunuzdan `OpenAI` uyumlu bir uç nokta (endpoint) olarak bağlanın.

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
                       api_base="http://localhost:7501/v1",  # bunun portunuza işaret ettiğinden emin olun
                       api_key="local", model_type="chat")
          dspy.configure(lm=lm)
          ```

    === "Diğer sağlayıcılar"
        DSPy'de, [LiteLLM tarafından desteklenen düzinelerce LLM sağlayıcısından](https://docs.litellm.ai/docs/providers) herhangi birini kullanabilirsiniz. Hangi `{PROVIDER}_API_KEY` değişkenini ayarlamanız ve yapıcıya (constructor) `{provider_name}/{model_name}` bilgisini nasıl iletmeniz gerektiği konusundaki talimatlarını takip etmeniz yeterlidir.

        Bazı örnekler:

        - `ANYSCALE_API_KEY` ile `anyscale/mistralai/Mistral-7B-Instruct-v0.1`
        - `TOGETHERAI_API_KEY` ile `together_ai/togethercomputer/llama-2-70b-chat`
        - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` ve `AWS_REGION_NAME` ile `sagemaker/<uç-nokta-adınız>`
        - `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` ve isteğe bağlı `AZURE_AD_TOKEN` ile `AZURE_API_TYPE` ile `azure/<dağıtım_adınız>`

        
        Sağlayıcınız OpenAI uyumlu bir uç nokta sunuyorsa, tam model adınızın başına bir `openai/` öneki eklemeniz yeterlidir.

        ```python linenums="1"
        import dspy
        lm = dspy.LM("openai/your-model-name", api_key="PROVIDER_API_KEY", api_base="YOUR_PROVIDER_URL")
        dspy.configure(lm=lm)
        ```

??? "LM'yi doğrudan çağırma."

     İdeomatik DSPy kullanımı, bu sayfanın geri kalanında tanımlayacağımız _modülleri_ (modules) kullanmayı içerir. Ancak, yukarıda yapılandırdığınız `lm` nesnesini doğrudan çağırmak da hala kolaydır. Bu size birleşik bir API sunar ve otomatik önbelleğe alma (caching) gibi yardımcı araçlardan yararlanmanızı sağlar.

     ```python linenums="1"       
     lm("Bunun bir test olduğunu söyle!", temperature=0.7)  # => ['Bu bir testtir!']
     lm(messages=[{"role": "user", "content": "Bunun bir test olduğunu söyle!"}])  # => ['Bu bir testtir!']
     ```


## 1) **Modüller**, yapay zeka davranışını metin dizileri olarak değil, _kod_ olarak tanımlamanıza yardımcı olur.

Güvenilir yapay zeka sistemleri oluşturmak için hızlıca yineleme yapmanız gerekir. Ancak istemleri (prompts) yönetmek bunu zorlaştırır: **LM'nizi, metriklerinizi veya boru hattınızı her değiştirdiğinizde** sizi metin dizileriyle veya verilerle uğraşmaya zorlar. 2020'den bu yana düzinelerce sınıfının en iyisi bileşik LM sistemi oluşturmuş bir ekip olarak bunu zor yoldan öğrendik ve bu nedenle yapay zeka sistemi tasarımını, belirli LM'ler veya istem stratejileri hakkındaki karmaşık tesadüfi seçimlerden ayırmak için DSPy'yi geliştirdik.

DSPy, odağınızı istem dizileriyle uğraşmaktan **yapılandırılmış ve bildirimsel doğal dil modülleriyle programlamaya** kaydırır. Sisteminizdeki her yapay zeka bileşeni için, girdi/çıktı davranışını bir _imza_ (signature) olarak belirtir ve LM'nizi çağırma stratejisi atamak için bir _modül_ seçersiniz. DSPy, imzalarınızı istemlere dönüştürür ve tiplendirilmiş çıktılarınızı ayrıştırır; böylece farklı modülleri ergonomik, taşınabilir ve optimize edilebilir yapay zeka sistemleri halinde bir araya getirebilirsiniz.

!!! info "Başlarken II: Çeşitli görevler için DSPy modülleri oluşturun"
    Yukarıda `lm` yapılandırmanızı yaptıktan sonra aşağıdaki örnekleri deneyin. LM'nizin kutudan çıktığı haliyle hangi görevleri iyi yapabildiğini keşfetmek için alanları (fields) özelleştirin. Aşağıdaki her sekme, göreve özel bir _imza_ ile `dspy.Predict`, `dspy.ChainOfThought` veya `dspy.ReAct` gibi bir DSPy modülü kurar. Örneğin, `question -> answer: float` imzası, modüle bir soru almasını ve `float` (ondalık sayı) türünde bir cevap üretmesini söyler.

    === "Matematik"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="İki zar atılıyor. Toplamın iki olma olasılığı nedir?")
        ```
        
        **Olası Çıktı:**
        ```text
        Prediction(
            reasoning='İki zar atıldığında, her zarın 6 yüzü vardır ve bu da toplam 6 x 6 = 36 olası sonuç doğurur. İki zarın üzerindeki sayıların toplamı, yalnızca her iki zar da 1 gösterdiğinde ikiye eşit olur. Bu sadece bir özel sonuçtur: (1, 1). Bu nedenle, sadece 1 uygun sonuç vardır. Toplamın iki olma olasılığı, uygun sonuç sayısının toplam olası sonuç sayısına bölünmesidir, bu da 1/36 eder.',
            answer=0.0277776
        )
        ```

    === "RAG (Geri Getirme Destekli Üretim)"

        ```python linenums="1"       
        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url="[http://20.102.90.50:2017/wiki17_abstracts](http://20.102.90.50:2017/wiki17_abstracts)")(query, k=3)
            return [x["text"] for x in results]
        
        rag = dspy.ChainOfThought("context, question -> response")

        question = "David Gregory'nin miras aldığı kalenin adı nedir?"
        rag(context=search_wikipedia(question), question=question)
        ```
        
        **Olası Çıktı:**
        ```text
        Prediction(
            reasoning='Bağlam, İskoç bir doktor ve mucit olan David Gregory hakkında bilgi sağlıyor. Özellikle 1664 yılında Kinnairdy Kalesi'ni miras aldığından bahsediyor. Bu ayrıntı, David Gregory'nin miras aldığı kalenin adıyla ilgili soruyu doğrudan yanıtlıyor.',
            response='Kinnairdy Kalesi'
        )
        ```

    === "Sınıflandırma"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Bir cümlenin duygu durumunu sınıflandırın."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="Son bölümü olmasa da bu kitabı okumak çok eğlenceliydi.")
        ```
        
        **Olası Çıktı:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Bilgi Çıkarımı"

        ```python linenums="1"         
        class ExtractInfo(dspy.Signature):
            """Metinden yapılandırılmış bilgi çıkarın."""
            
            text: str = dspy.InputField()
            title: str = dspy.OutputField()
            headings: list[str] = dspy.OutputField()
            entities: list[dict[str, str]] = dspy.OutputField(desc="varlıkların ve meta verilerinin bir listesi")
        
        module = dspy.Predict(ExtractInfo)

        text = "Apple Inc. bugün en yeni iPhone 14'ü duyurdu." \
            "CEO Tim Cook, bir basın açıklamasında yeni özelliklerin altını çizdi."
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities)
        ```
        
        **Olası Çıktı:**
        ```text
        Apple Inc. iPhone 14'ü Duyurdu
        ['Giriş', "CEO'nun Açıklaması", 'Yeni Özellikler']
        [{'name': 'Apple Inc.', 'type': 'Organization'}, {'name': 'iPhone 14', 'type': 'Product'}, {'name': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Ajanlar (Agents)"

        ```python linenums="1"       
        def evaluate_math(expression: str):
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str):
            results = dspy.ColBERTv2(url="[http://20.102.90.50:2017/wiki17_abstracts](http://20.102.90.50:2017/wiki17_abstracts)")(query, k=3)
            return [x["text"] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="9362158'in Kinnairdy kalesinden David Gregory'nin doğum yılına bölümü kaçtır?")
        print(pred.answer)
        ```
        
        **Olası Çıktı:**

        ```text
        5761.328
        ```
    
    === "Çok Aşamalı Boru Hatları"

        ```python linenums="1"       
        class Outline(dspy.Signature):
            """Bir konunun kapsamlı bir taslağını çıkarın."""
            
            topic: str = dspy.InputField()
            title: str = dspy.OutputField()
            sections: list[str] = dspy.OutputField()
            section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="bölüm başlıklarından alt başlıklara eşleme")

        class DraftSection(dspy.Signature):
            """Bir makalenin üst düzey bir bölümünü taslak haline getirin."""
            
            topic: str = dspy.InputField()
            section_heading: str = dspy.InputField()
            section_subheadings: list[str] = dspy.InputField()
            content: str = dspy.OutputField(desc="markdown formatında bölüm")

        class DraftArticle(dspy.Module):
            def __init__(self):
                self.build_outline = dspy.ChainOfThought(Outline)
                self.draft_section = dspy.ChainOfThought(DraftSection)

            def forward(self, topic):
                outline = self.build_outline(topic=topic)
                sections = []
                for heading, subheadings in outline.section_subheadings.items():
                    section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
                    section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
                    sections.append(section.content)
                return dspy.Prediction(title=outline.title, sections=sections)

        draft_article = DraftArticle()
        article = draft_article(topic="2002 Dünya Kupası")
        ```
        
        **Olası Çıktı:**

        Konu üzerine 1500 kelimelik bir makale, örneğin:

        ```text
        ## Eleme Süreci

        2002 FIFA Dünya Kupası eleme süreci bir dizi..... [sunum için burada kısaltılmıştır].

        ### UEFA Elemeleri

        UEFA elemeleri, 13 koltuk için yarışan 50 takımı içeriyordu..... [sunum için burada kısaltılmıştır].

        .... [makalenin geri kalanı]
        ```

        DSPy'nin bunun gibi çok aşamalı modülleri optimize etmeyi kolaylaştırdığını unutmayın. Sistemin *nihai* çıktısını değerlendirebildiğiniz sürece, her DSPy optimize edici tüm ara modülleri ince ayarlayabilir (tune).

??? "Pratikte DSPy kullanımı: hızlı betik yazımından karmaşık sistemler oluşturmaya."

    Standart istemler (prompts), arayüzü ("LM ne yapmalı?") uygulama yöntemiyle ("bunu yapmasını ona nasıl söyleriz?") birleştirir. DSPy, arayüzü _imzalar_ (signatures) olarak ayrıştırır; böylece uygulama yöntemini, daha büyük bir program bağlamında veriden çıkarabilir veya öğrenebiliriz.
    
    Optimize edicileri kullanmaya başlamadan önce bile, DSPy modülleri etkili LM sistemlerini ergonomik ve taşınabilir _kodlar_ olarak yazmanıza olanak tanır. Birçok görev ve LM genelinde, yerleşik DSPy adaptörlerinin güvenilirliğini değerlendiren _imza test paketleri_ (signature test suites) bulunduruyoruz. Adaptörler, optimizasyon öncesinde imzaları istemlere eşleyen bileşenlerdir. Eğer LM'niz için basit bir istemin, ideomatik DSPy kullanımından sürekli olarak daha iyi performans gösterdiği bir görev bulursanız, bunu bir hata olarak kabul edin ve bir [sorun kaydı (issue) oluşturun](https://github.com/stanfordnlp/dspy/issues). Bunu yerleşik adaptörleri geliştirmek için kullanacağız.


## 2) **Optimize Ediciler (Optimizers)**, yapay zeka modüllerinizin istemlerini ve ağırlıklarını ayarlar.

DSPy, doğal dil açıklamaları içeren üst düzey kodları; LM'nizi programınızın yapısı ve metrikleriyle uyumlu hale getiren alt düzey hesaplamalara, istemlere veya ağırlık güncellemelerine derlemek için gereken araçları sağlar. Kodunuzu veya metriklerinizi değiştirirseniz, buna göre yeniden derleme yapmanız yeterlidir.

Görevinizi temsil eden birkaç on veya yüzlerce _girdi_ örneği ve sisteminizin çıktılarının kalitesini ölçebilen bir _metrik_ sağlandığında, bir DSPy optimize edici kullanabilirsiniz. DSPy'deki farklı optimize ediciler; `dspy.BootstrapRS`<sup>[1](https://arxiv.org/abs/2310.03714)</sup> gibi her modül için **iyi az-örnekli (few-shot) örnekler sentezleyerek**, `dspy.GEPA`<sup>[2](https://arxiv.org/abs/2507.19457)</sup>, `dspy.MIPROv2`<sup>[3](https://arxiv.org/abs/2406.11695)</sup> gibi her istem için **daha iyi doğal dil talimatları önerip akıllıca keşfederek** ve `dspy.BootstrapFinetune`<sup>[4](https://arxiv.org/abs/2407.10930)</sup> gibi **modülleriniz için veri kümeleri oluşturup bunları sisteminizdeki LM ağırlıklarına ince ayar (finetune) yapmak için kullanarak** çalışır. `dspy.GEPA` kullanımıyla ilgili ayrıntılı eğitimler için lütfen [dspy.GEPA eğitimlerine](https://dspy.ai/tutorials/gepa_ai_program/) göz atın.


!!! info "Başlarken III: DSPy programlarında LM istemlerini veya ağırlıklarını optimize etme"
    Tipik bir basit optimizasyon çalışması yaklaşık 2 USD tutar ve yaklaşık 20 dakika sürer; ancak çok büyük LM'ler veya çok büyük veri kümeleriyle optimize edicileri çalıştırırken dikkatli olun.
    Optimizasyon maliyeti; LM'nize, veri kümenize ve yapılandırmanıza bağlı olarak birkaç sentten onlarca dolara kadar değişebilir.

    Aşağıdaki örnekler `HuggingFace/datasets` kütüphanesine dayanmaktadır, aşağıdaki komutla kurulumunu yapabilirsiniz.

    ```bash
    > pip install -U datasets
    ```

    === "Bir ReAct ajanı için istemleri optimize etme"
        Bu, Wikipedia üzerinden arama yaparak soruları yanıtlayan bir `dspy.ReAct` ajanı kurmanın ve ardından `HotPotQA` veri kümesinden örneklenen 500 soru-cevap çifti üzerinde ekonomik `light` modunda `dspy.MIPROv2` kullanarak onu optimize etmenin minimal ama tamamen çalıştırılabilir bir örneğidir.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url="[http://20.102.90.50:2017/wiki17_abstracts](http://20.102.90.50:2017/wiki17_abstracts)")(query, k=3)
            return [x["text"] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        Bunun gibi gayriresmi bir çalışma, `gpt-4o-mini` modeline görevin detayları hakkında daha fazla şey öğreterek ReAct'in puanını %24'ten %51'e çıkarır.

    === "RAG için istemleri optimize etme"
        Arama yapılacak (`search`) bir geri getirme indeksi, favori `dspy.LM` modeliniz ve sorulardan oluşan küçük bir `trainset` (eğitim kümesi) verildiğinde; aşağıdaki kod parçası, uzun çıktılı RAG sisteminizi, bir DSPy modülü olarak uygulanan yerleşik `SemanticF1` metriğine göre optimize edebilir.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought("context, question -> response")

            def forward(self, question):
                context = search(question, k=self.num_docs)   # aşağıdaki eğitimde tanımlanmıştır
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.evaluate.SemanticF1(decompositional=True), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        Çalıştırabileceğiniz eksiksiz bir RAG örneği için bu [eğitime](tutorials/rag/index.ipynb) başlayın. Bu eğitim, StackExchange topluluklarının bir alt kümesi üzerindeki RAG sisteminin kalitesini %10 göreceli kazançla artırır.

    === "Sınıflandırma için ağırlıkları optimize etme"
        <details><summary>Veri kümesi kurulum kodunu göstermek için tıklayın.</summary>

        ```python linenums="1"
        import random
        from typing import Literal

        from datasets import load_dataset

        import dspy
        from dspy.datasets import DataLoader

        # Banking77 veri kümesini yükleyin.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
        kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

        # Veri kümesinden ilk 2000 örneği yükleyin ve her *eğitim* örneğine bir ipucu (hint) atayın.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        lm=dspy.LM('openai/gpt-4o-mini-2024-07-18')

        # Sınıflandırma için DSPy modülünü tanımlayın. Varsa, eğitim sırasında ipucunu kullanacaktır.
        signature = dspy.Signature("text, hint -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThought(signature)
        classify.set_lm(lm)

        # BootstrapFinetune aracılığıyla optimize edin.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="Bekleyen bir nakit çekme işlemi ne anlama gelir?")
        
        # Tam bir ince ayar (fine-tuning) eğitimi için bkz: [https://dspy.ai/tutorials/classification_finetuning/](https://dspy.ai/tutorials/classification_finetuning/)
        ```

        **Olası Çıktı (son satırdan):**
        ```text
        Prediction(
            reasoning='Bekleyen bir nakit çekme işlemi, nakit çekme talebinin başlatıldığını ancak henüz tamamlanmadığını veya işlenmediğini gösterir. Bu durum, işlemin hala devam ettiği ve fonların henüz hesaptan düşülmediği veya kullanıcıya sunulmadığı anlamına gelir.',
            label='pending_cash_withdrawal'
        )
        ```

        DSPy 2.5.29 üzerinde buna benzer gayriresmi bir çalışma, GPT-4o-mini'nin puanını %66'dan %87'ye çıkarır.


??? "Bir DSPy optimize edici örneği nedir? Farklı optimize ediciler nasıl çalışır?"

    `dspy.MIPROv2` optimize edicisini örnek olarak alalım. İlk olarak, MIPRO **önyükleme aşaması (bootstrapping stage)** ile başlar. Bu noktada optimize edilmemiş olabilecek programınızı alır ve her bir modülünüz için girdi/çıktı davranışı izlerini toplamak amacıyla farklı girdiler üzerinde defalarca çalıştırır. Bu izleri, metriğiniz tarafından yüksek puan alan yörüngelerde görünenleri tutacak şekilde filtreler. İkinci olarak, MIPRO **temellendirilmiş öneri aşamasına (grounded proposal stage)** girer. DSPy programınızın kodunu, verilerinizi ve programınızı çalıştırmadan elde edilen izleri önizler; bunları programınızdaki her bir istem için birçok potansiyel talimat taslağı hazırlamak için kullanır. Üçüncü olarak, MIPRO **ayrık arama aşamasını (discrete search stage)** başlatır. Eğitim kümenizden mini gruplar (mini-batches) örnekler, boru hattındaki her bir istemi oluşturmak için kullanılacak talimat ve izlerin bir kombinasyonunu önerir ve aday programı mini grup üzerinde değerlendirir. Elde edilen puanı kullanan MIPRO, önerilerin zamanla daha iyi hale gelmesine yardımcı olan bir vekil modeli (surrogate model) günceller.

    DSPy optimize edicilerini bu kadar güçlü kılan şeylerden biri de birleştirilebilir olmalarıdır. `dspy.MIPROv2`'yi çalıştırabilir ve üretilen programı tekrar `dspy.MIPROv2`'ye veya örneğin daha iyi sonuçlar almak için `dspy.BootstrapFinetune`'a girdi olarak verebilirsiniz. Bu, kısmen `dspy.BetterTogether` yaklaşımının özüdür. Alternatif olarak, optimize ediciyi çalıştırıp ardından en iyi 5 aday programı çıkarabilir ve bunlardan bir `dspy.Ensemble` (topluluk) oluşturabilirsiniz. Bu, hem *çıkarım zamanı hesaplamasını* (örneğin topluluklar) hem de DSPy'nin benzersiz *çıkarım öncesi zaman hesaplamasını* (yani optimizasyon bütçesi) son derece sistematik yollarla ölçeklendirmenize olanak tanır.


## 3) **DSPy Ekosistemi** açık kaynaklı yapay zeka araştırmalarını ileriye taşıyor.

Monolitik LM'lerle karşılaştırıldığında, DSPy'nin modüler paradigması büyük bir topluluğun LM programları için birleşimsel mimarileri, çıkarım zamanı stratejilerini ve optimize edicileri açık ve dağıtık bir şekilde geliştirmesine olanak tanır. Bu, DSPy kullanıcılarına daha fazla kontrol sağlar, çok daha hızlı yineleme yapmalarına yardımcı olur ve en son optimize edicileri veya modülleri uygulayarak programlarının zamanla daha iyi hale gelmesini sağlar.

DSPy araştırma çabası, [ColBERT-QA](https://arxiv.org/abs/2007.00814), [Baleen](https://arxiv.org/abs/2101.00436) ve [Hindsight](https://arxiv.org/abs/2110.07752) gibi erken dönem [bileşik LM sistemlerini](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) geliştirirken öğrendiklerimiz üzerine inşa edilerek Şubat 2022'de Stanford NLP'de başladı. İlk versiyon Aralık 2022'de [DSP](https://arxiv.org/abs/2212.14024) olarak yayınlandı ve Ekim 2023'te [DSPy](https://arxiv.org/abs/2310.03714) haline geldi. [250 katkıda bulunmacı](https://github.com/stanfordnlp/dspy/graphs/contributors) sayesinde DSPy, yüz binlerce insanı modüler LM programları oluşturma ve optimize etme ile tanıştırdı.

O zamandan beri DSPy topluluğu; [MIPROv2](https://arxiv.org/abs/2406.11695), [BetterTogether](https://arxiv.org/abs/2407.10930) ve [LeReT](https://arxiv.org/abs/2410.23214) gibi optimize ediciler; [STORM](https://arxiv.org/abs/2402.14207), [IReRa](https://arxiv.org/abs/2401.12178) ve [DSPy Assertions](https://arxiv.org/abs/2312.13382) gibi program mimarileri; ve [PAPILLON](https://arxiv.org/abs/2410.17127), [PATH](https://arxiv.org/abs/2406.11706), [WangLab@MEDIQA](https://arxiv.org/abs/2404.14544), [UMD'nin İstem Çalışması](https://arxiv.org/abs/2406.06608) ve [Haize'nin Red-Teaming Programı](https://blog.haizelabs.com/posts/dspy/) gibi yeni problemlere başarılı uygulamaların yanı sıra birçok açık kaynaklı proje, üretim uygulaması ve diğer [kullanım durumları](community/use-cases.md) üzerinde geniş bir çalışma külliyatı üretti.