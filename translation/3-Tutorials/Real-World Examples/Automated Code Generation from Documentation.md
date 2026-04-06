# DSPy ile Dokümantasyondan Otomatik Kod Üretimi

Bu eğitim, DSPy kullanarak URL’lerden dokümantasyonu otomatik olarak nasıl çekebileceğinizi ve herhangi bir kütüphane için çalışan kod örnekleri nasıl üretebileceğinizi gösterir. Sistem, dokümantasyon sitelerini analiz edebilir, temel kavramları çıkarabilir ve amaca uygun kod örnekleri üretebilir. fileciteturn32file0

## Ne İnşa Edeceksiniz

Şunları yapabilen, dokümantasyon destekli bir kod üretim sistemi:

- Birden fazla URL’den dokümantasyonu çeker ve ayrıştırır
- API kalıplarını, metotları ve kullanım örneklerini çıkarır  
- Belirli kullanım senaryoları için çalışan kod üretir
- Açıklamalar ve en iyi uygulamaları sunar
- Herhangi bir kütüphanenin dokümantasyonuyla çalışır fileciteturn32file0

## Kurulum

```bash
pip install dspy requests beautifulsoup4 html2text
```

## Adım 1: Dokümantasyon Çekme ve İşleme

```python
import dspy
import requests
from bs4 import BeautifulSoup
import html2text
from typing import List, Dict, Any
import json
from urllib.parse import urljoin, urlparse
import time

# DSPy'yi yapılandır
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

class DocumentationFetcher:
    """URL'lerden dokümantasyon çeker ve işler."""

    def __init__(self, max_retries=3, delay=1):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.max_retries = max_retries
        self.delay = delay
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True

    def fetch_url(self, url: str) -> dict[str, str]:
        """Tek bir URL'den içerik çek."""
        for attempt in range(self.max_retries):
            try:
                print(f"📡 Çekiliyor: {url} (deneme {attempt + 1})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Script ve stil öğelerini kaldır
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Daha iyi LLM işleme için markdown'a dönüştür
                markdown_content = self.html_converter.handle(str(soup))

                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "Başlık yok",
                    "content": markdown_content,
                    "success": True
                }

            except Exception as e:
                print(f"❌ {url} çekilirken hata: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay)
                else:
                    return {
                        "url": url,
                        "title": "Çekilemedi",
                        "content": f"Hata: {str(e)}",
                        "success": False
                    }

        return {"url": url, "title": "Başarısız", "content": "", "success": False}

    def fetch_documentation(self, urls: list[str]) -> list[dict[str, str]]:
        """Birden fazla URL'den dokümantasyon çek."""
        results = []

        for url in urls:
            result = self.fetch_url(url)
            results.append(result)
            time.sleep(self.delay)  # Sunuculara saygılı ol

        return results

class LibraryAnalyzer(dspy.Signature):
    """Temel kavramları ve kalıpları anlamak için kütüphane dokümantasyonunu analiz et."""
    library_name: str = dspy.InputField(desc="Analiz edilecek kütüphanenin adı")
    documentation_content: str = dspy.InputField(desc="Birleştirilmiş dokümantasyon içeriği")

    core_concepts: list[str] = dspy.OutputField(desc="Ana kavramlar ve bileşenler")
    common_patterns: list[str] = dspy.OutputField(desc="Yaygın kullanım kalıpları")
    key_methods: list[str] = dspy.OutputField(desc="Önemli metotlar ve fonksiyonlar")
    installation_info: str = dspy.OutputField(desc="Kurulum ve hazırlık bilgileri")
    code_examples: list[str] = dspy.OutputField(desc="Bulunan örnek kod parçaları")

class CodeGenerator(dspy.Signature):
    """Hedef kütüphaneyi kullanarak belirli kullanım senaryoları için kod örnekleri üret."""
    library_info: str = dspy.InputField(desc="Kütüphane kavramları ve kalıpları")
    use_case: str = dspy.InputField(desc="Uygulanacak belirli kullanım senaryosu")
    requirements: str = dspy.InputField(desc="Ek gereksinimler veya kısıtlar")

    code_example: str = dspy.OutputField(desc="Eksiksiz, çalışan kod örneği")
    explanation: str = dspy.OutputField(desc="Kodun adım adım açıklaması")
    best_practices: list[str] = dspy.OutputField(desc="En iyi uygulamalar ve ipuçları")
    imports_needed: list[str] = dspy.OutputField(desc="Gerekli importlar ve bağımlılıklar")

class DocumentationLearningAgent(dspy.Module):
    """Dokümantasyon URL'lerinden öğrenen ve kod örnekleri üreten ajan."""

    def __init__(self):
        super().__init__()
        self.fetcher = DocumentationFetcher()
        self.analyze_docs = dspy.ChainOfThought(LibraryAnalyzer)
        self.generate_code = dspy.ChainOfThought(CodeGenerator)
        self.refine_code = dspy.ChainOfThought(
            "code, feedback -> improved_code: str, changes_made: list[str]"
        )

    def learn_from_urls(self, library_name: str, doc_urls: list[str]) -> Dict:
        """Bir kütüphane hakkında dokümantasyon URL'lerinden öğren."""

        print(f"📚 {library_name} hakkında {len(doc_urls)} URL'den öğreniliyor...")

        # Tüm dokümantasyonu çek
        docs = self.fetcher.fetch_documentation(doc_urls)

        # Başarıyla çekilenleri birleştir
        combined_content = "\n\n---\n\n".join([
            f"URL: {doc['url']}\nTitle: {doc['title']}\n\n{doc['content']}"
            for doc in docs if doc['success']
        ])

        if not combined_content:
            raise ValueError("Hiçbir dokümantasyon başarıyla çekilemedi")

        # Birleştirilmiş dokümantasyonu analiz et
        analysis = self.analyze_docs(
            library_name=library_name,
            documentation_content=combined_content
        )

        return {
            "library": library_name,
            "source_urls": [doc['url'] for doc in docs if doc['success']],
            "core_concepts": analysis.core_concepts,
            "patterns": analysis.common_patterns,
            "methods": analysis.key_methods,
            "installation": analysis.installation_info,
            "examples": analysis.code_examples,
            "fetched_docs": docs
        }

    def generate_example(self, library_info: Dict, use_case: str, requirements: str = "") -> Dict:
        """Belirli bir kullanım senaryosu için bir kod örneği üret."""

        # Üretici için kütüphane bilgisini biçimlendir
        info_text = f"""
        Library: {library_info['library']}
        Core Concepts: {', '.join(library_info['core_concepts'])}
        Common Patterns: {', '.join(library_info['patterns'])}
        Key Methods: {', '.join(library_info['methods'])}
        Installation: {library_info['installation']}
        Example Code Snippets: {'; '.join(library_info['examples'][:3])}  # İlk 3 örnek
        """

        code_result = self.generate_code(
            library_info=info_text,
            use_case=use_case,
            requirements=requirements
        )

        return {
            "code": code_result.code_example,
            "explanation": code_result.explanation,
            "best_practices": code_result.best_practices,
            "imports": code_result.imports_needed
        }

# Öğrenme ajanını başlat
agent = DocumentationLearningAgent()
```

## Adım 2: Dokümantasyon URL'lerinden Öğrenme

```python
def learn_library_from_urls(library_name: str, documentation_urls: list[str]) -> Dict:
    """Herhangi bir kütüphane hakkında dokümantasyon URL'lerinden öğren."""

    try:
        library_info = agent.learn_from_urls(library_name, documentation_urls)

        print(f"\n🔍 {library_name} için Kütüphane Analizi Sonuçları:")
        print(f"Kaynaklar: {len(library_info['source_urls'])} başarılı çekim")
        print(f"Temel Kavramlar: {library_info['core_concepts']}")
        print(f"Yaygın Kalıplar: {library_info['patterns']}")
        print(f"Önemli Metotlar: {library_info['methods']}")
        print(f"Kurulum: {library_info['installation']}")
        print(f"{len(library_info['examples'])} kod örneği bulundu")

        return library_info

    except Exception as e:
        print(f"❌ Kütüphane öğrenilirken hata: {e}")
        raise

# Örnek 1: FastAPI'yi resmi dokümantasyondan öğren
fastapi_urls = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/"
]

print("🚀 FastAPI resmi dokümantasyondan öğreniliyor...")
fastapi_info = learn_library_from_urls("FastAPI", fastapi_urls)

# Örnek 2: Farklı bir kütüphaneyi öğren (bunu istediğiniz kütüphaneyle değiştirebilirsiniz)
streamlit_urls = [
    "https://docs.streamlit.io/",
    "https://docs.streamlit.io/get-started",
    "https://docs.streamlit.io/develop/api-reference"
]

print("\n\n📊 Streamlit resmi dokümantasyondan öğreniliyor...")
streamlit_info = learn_library_from_urls("Streamlit", streamlit_urls)
```

## Adım 3: Kod Örnekleri Üretme

```python
def generate_examples_for_library(library_info: Dict, library_name: str):
    """Dokümantasyonuna dayanarak herhangi bir kütüphane için kod örnekleri üret."""

    # Çoğu kütüphaneye uygulanabilecek genel kullanım senaryolarını tanımla
    use_cases = [
        {
            "name": "Temel Kurulum ve Hello World",
            "description": f"{library_name} ile en küçük çalışan örneği oluştur",
            "requirements": "Kurulum, importlar ve temel kullanımı ekle"
        },
        {
            "name": "Yaygın İşlemler",
            "description": f"En yaygın {library_name} işlemlerini göster",
            "requirements": "Tipik iş akışını ve en iyi uygulamaları göster"
        },
        {
            "name": "İleri Seviye Kullanım",
            "description": f"{library_name} yeteneklerini sergileyen daha karmaşık bir örnek oluştur",
            "requirements": "Hata yönetimi ve optimizasyon ekle"
        }
    ]

    generated_examples = []

    print(f"\n🔧 {library_name} için örnekler üretiliyor...")

    for use_case in use_cases:
        print(f"\n📝 {use_case['name']}")
        print(f"Açıklama: {use_case['description']}")

        example = agent.generate_example(
            library_info=library_info,
            use_case=use_case['description'],
            requirements=use_case['requirements']
        )

        print("\n💻 Üretilen Kod:")
        print("```python")
        print(example['code'])
        print("```")

        print("\n📦 Gerekli Importlar:")
        for imp in example['imports']:
            print(f"  • {imp}")

        print("\n📝 Açıklama:")
        print(example['explanation'])

        print("\n✅ En İyi Uygulamalar:")
        for practice in example['best_practices']:
            print(f"  • {practice}")

        generated_examples.append({
            "use_case": use_case['name'],
            "code": example['code'],
            "imports": example['imports'],
            "explanation": example['explanation'],
            "best_practices": example['best_practices']
        })

        print("-" * 80)

    return generated_examples

# Her iki kütüphane için de örnekler üret
print("🎯 FastAPI Örnekleri Üretiliyor:")
fastapi_examples = generate_examples_for_library(fastapi_info, "FastAPI")

print("\n\n🎯 Streamlit Örnekleri Üretiliyor:")
streamlit_examples = generate_examples_for_library(streamlit_info, "Streamlit")
```

## Adım 4: Etkileşimli Kütüphane Öğrenme Fonksiyonu

```python
def learn_any_library(library_name: str, documentation_urls: list[str], use_cases: list[str] = None):
    """Dokümantasyonundan herhangi bir kütüphaneyi öğren ve örnekler üret."""

    if use_cases is None:
        use_cases = [
            "Temel kurulum ve hello world örneği",
            "Yaygın işlemler ve iş akışları",
            "En iyi uygulamalarla ileri seviye kullanım"
        ]

    print(f"🚀 {library_name} için otomatik öğrenme başlatılıyor...")
    print(f"Dokümantasyon kaynakları: {len(documentation_urls)} URL")

    try:
        # Adım 1: Dokümantasyondan öğren
        library_info = agent.learn_from_urls(library_name, documentation_urls)

        # Adım 2: Her kullanım senaryosu için örnekler üret
        all_examples = []

        for i, use_case in enumerate(use_cases, 1):
            print(f"\n📝 Örnek {i}/{len(use_cases)} üretiliyor: {use_case}")

            example = agent.generate_example(
                library_info=library_info,
                use_case=use_case,
                requirements="Hata yönetimi, yorumlar ekle ve en iyi uygulamalara uy"
            )

            all_examples.append({
                "use_case": use_case,
                "code": example['code'],
                "imports": example['imports'],
                "explanation": example['explanation'],
                "best_practices": example['best_practices']
            })

        return {
            "library_info": library_info,
            "examples": all_examples
        }

    except Exception as e:
        print(f"❌ {library_name} öğrenilirken hata: {e}")
        return None

def interactive_learning_session():
    """Kullanıcı girdisiyle kütüphane öğrenmek için etkileşimli oturum."""

    print("🎯 Etkileşimli Kütüphane Öğrenme Sistemine Hoş Geldiniz!")
    print("Bu sistem, herhangi bir Python kütüphanesini dokümantasyonundan öğrenmenize yardımcı olacak.\n")

    learned_libraries = {}

    while True:
        print("\n" + "="*60)
        print("🚀 KÜTÜPHANE ÖĞRENME OTURUMU")
        print("="*60)

        # Kullanıcıdan kütüphane adını al
        library_name = input("\n📚 Öğrenmek istediğiniz kütüphane adını girin (çıkmak için 'quit'): ").strip()

        if library_name.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Etkileşimli Kütüphane Öğrenme Sistemini kullandığınız için teşekkürler!")
            break

        if not library_name:
            print("❌ Lütfen geçerli bir kütüphane adı girin.")
            continue

        # Dokümantasyon URL'lerini al
        print(f"\n🔗 {library_name} için dokümantasyon URL'lerini girin (satır başına bir URL, bitirmek için boş satır):")
        urls = []
        while True:
            url = input("  URL: ").strip()
            if not url:
                break
            if not url.startswith(('http://', 'https://')):
                print("    ⚠️  Lütfen http:// veya https:// ile başlayan geçerli bir URL girin")
                continue
            urls.append(url)

        if not urls:
            print("❌ Geçerli URL girilmedi. Bu kütüphane atlanıyor.")
            continue

        # Kullanıcıdan özel kullanım senaryolarını al
        print(f"\n🎯 {library_name} için kullanım senaryolarını tanımlayın (isteğe bağlı, varsayılanlar için Enter'a basın):")
        print("   Varsayılan kullanım senaryoları: Temel kurulum, Yaygın işlemler, İleri seviye kullanım")

        user_wants_custom = input("   Özel kullanım senaryoları tanımlamak istiyor musunuz? (e/h): ").strip().lower()

        use_cases = None
        if user_wants_custom in ['e', 'evet', 'y', 'yes']:
            print("   Kullanım senaryolarınızı girin (satır başına bir tane, bitirmek için boş satır):")
            use_cases = []
            while True:
                use_case = input("     Kullanım senaryosu: ").strip()
                if not use_case:
                    break
                use_cases.append(use_case)

            if not use_cases:
                print("   Özel kullanım senaryosu girilmedi, varsayılanlar kullanılacak.")
                use_cases = None

        # Kütüphaneyi öğren
        print(f"\n🚀 {library_name} için öğrenme süreci başlatılıyor...")
        result = learn_any_library(library_name, urls, use_cases)

        if result:
            learned_libraries[library_name] = result
            print(f"\n✅ {library_name} başarıyla öğrenildi!")

            # Özeti göster
            print(f"\n📊 {library_name} için Öğrenme Özeti:")
            print(f"   • Temel kavramlar: {len(result['library_info']['core_concepts'])} tane belirlendi")
            print(f"   • Yaygın kalıplar: {len(result['library_info']['patterns'])} tane bulundu")
            print(f"   • Üretilen örnekler: {len(result['examples'])}")

            # Kullanıcının örnekleri görmek isteyip istemediğini sor
            show_examples = input(f"\n👀 {library_name} için üretilen örnekleri görmek ister misiniz? (e/h): ").strip().lower()

            if show_examples in ['e', 'evet', 'y', 'yes']:
                for i, example in enumerate(result['examples'], 1):
                    print(f"\n{'─'*50}")
                    print(f"📝 Örnek {i}: {example['use_case']}")
                    print(f"{'─'*50}")

                    print("\n💻 Üretilen Kod:")
                    print("```python")
                    print(example['code'])
                    print("```")

                    print(f"\n📦 Gerekli Importlar:")
                    for imp in example['imports']:
                        print(f"  • {imp}")

                    print(f"\n📝 Açıklama:")
                    print(example['explanation'])

                    print(f"\n✅ En İyi Uygulamalar:")
                    for practice in example['best_practices']:
                        print(f"  • {practice}")

                    # Sonraki örneği görmek isteyip istemediğini sor
                    if i < len(result['examples']):
                        continue_viewing = input(f"\nSonraki örneğe geçilsin mi? (e/h): ").strip().lower()
                        if continue_viewing not in ['e', 'evet', 'y', 'yes']:
                            break

            # Sonuçları kaydetme teklifi
            save_results = input(f"\n💾 {library_name} için öğrenme sonuçları dosyaya kaydedilsin mi? (e/h): ").strip().lower()

            if save_results in ['e', 'evet', 'y', 'yes']:
                filename = input(f"   Dosya adını girin (varsayılan: {library_name.lower()}_learning.json): ").strip()
                if not filename:
                    filename = f"{library_name.lower()}_learning.json"

                try:
                    import json
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"   ✅ Sonuçlar {filename} dosyasına kaydedildi")
                except Exception as e:
                    print(f"   ❌ Dosya kaydedilirken hata: {e}")

        else:
            print(f"❌ {library_name} öğrenilemedi")

        # Başka bir kütüphane öğrenmek isteyip istemediğini sor
        print(f"\n📚 Şu ana kadar öğrenilen kütüphaneler: {list(learned_libraries.keys())}")
        continue_learning = input("\n🔄 Başka bir kütüphane öğrenmek ister misiniz? (e/h): ").strip().lower()

        if continue_learning not in ['e', 'evet', 'y', 'yes']:
            break

    # Son özet
    if learned_libraries:
        print(f"\n🎉 Oturum Özeti:")
        print(f"Başarıyla öğrenilen kütüphane sayısı: {len(learned_libraries)}")
        for lib_name, info in learned_libraries.items():
            print(f"  • {lib_name}: {len(info['examples'])} örnek üretildi")

    return learned_libraries

# Örnek: Etkileşimli öğrenme oturumunu çalıştır
if __name__ == "__main__":
    # Etkileşimli oturumu çalıştır
    learned_libraries = interactive_learning_session()
```

## Örnek Çıktı

Etkileşimli öğrenme sistemini çalıştırdığınızda şunları görürsünüz:

**Etkileşimli Oturum Başlangıcı:**
```
🎯 Etkileşimli Kütüphane Öğrenme Sistemine Hoş Geldiniz!
Bu sistem, herhangi bir Python kütüphanesini dokümantasyonundan öğrenmenize yardımcı olacak.

============================================================
🚀 KÜTÜPHANE ÖĞRENME OTURUMU
============================================================

📚 Öğrenmek istediğiniz kütüphane adını girin (çıkmak için 'quit'): FastAPI

🔗 FastAPI için dokümantasyon URL'lerini girin (satır başına bir URL, bitirmek için boş satır):
  URL: https://fastapi.tiangolo.com/
  URL: https://fastapi.tiangolo.com/tutorial/first-steps/
  URL: https://fastapi.tiangolo.com/tutorial/path-params/
  URL: 

🎯 FastAPI için kullanım senaryolarını tanımlayın (isteğe bağlı, varsayılanlar için Enter'a basın):
   Varsayılan kullanım senaryoları: Temel kurulum, Yaygın işlemler, İleri seviye kullanım
   Özel kullanım senaryoları tanımlamak istiyor musunuz? (y/n): y
   Kullanım senaryolarınızı girin (satır başına bir tane, bitirmek için boş satır):
     Kullanım senaryosu: Kimlik doğrulamalı bir REST API oluştur
     Kullanım senaryosu: Dosya yükleme endpoint'i geliştir
     Kullanım senaryosu: SQLAlchemy ile veritabanı entegrasyonu ekle
     Kullanım senaryosu: 
```

**Dokümantasyon İşleme:**
```
🚀 FastAPI için öğrenme süreci başlatılıyor...
🚀 FastAPI için otomatik öğrenme başlatılıyor...
Dokümantasyon kaynakları: 3 URL
📡 Çekiliyor: https://fastapi.tiangolo.com/ (deneme 1)
📡 Çekiliyor: https://fastapi.tiangolo.com/tutorial/first-steps/ (deneme 1)
📡 Çekiliyor: https://fastapi.tiangolo.com/tutorial/path-params/ (deneme 1)
📚 FastAPI hakkında 3 URL'den öğreniliyor...

🔍 FastAPI için Kütüphane Analizi Sonuçları:
Kaynaklar: 3 başarılı çekim
Temel Kavramlar: ['FastAPI app', 'path operations', 'dependencies', 'request/response models']
Yaygın Kalıplar: ['app = FastAPI()', 'decorator-based routing', 'Pydantic models']
Önemli Metotlar: ['FastAPI()', '@app.get()', '@app.post()', 'uvicorn.run()']
Kurulum: pip install fastapi uvicorn
```

**Kod Üretimi:**
```
📝 Örnek 1/3 üretiliyor: Kimlik doğrulamalı bir REST API oluştur

✅ FastAPI başarıyla öğrenildi!

📊 FastAPI için Öğrenme Özeti:
   • Temel kavramlar: 4 tane belirlendi
   • Yaygın kalıplar: 3 tane bulundu
   • Üretilen örnekler: 3

👀 FastAPI için üretilen örnekleri görmek ister misiniz? (y/n): y

──────────────────────────────────────────────────
📝 Örnek 1: Kimlik doğrulamalı bir REST API oluştur
──────────────────────────────────────────────────

💻 Üretilen Kod:
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from typing import Dict
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="Authenticated API", version="1.0.0")
security = HTTPBearer()

# JWT için gizli anahtar (production'da environment variable kullanın)
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Geçersiz token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Geçersiz token")

@app.post("/login")
async def login(username: str, password: str) -> dict[str, str]:
    # Production'da veritabanına karşı doğrulayın
    if username == "admin" and password == "secret":
        token_data = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=24)}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Geçersiz kimlik bilgileri")

@app.get("/protected")
async def protected_route(current_user: str = Depends(verify_token)) -> dict[str, str]:
    return {"message": f"Merhaba {current_user}! Burası korumalı bir rotadır."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

📦 Gerekli Importlar:
  • pip install fastapi uvicorn python-jose[cryptography]
  • from fastapi import FastAPI, Depends, HTTPException, status
  • from fastapi.security import HTTPBearer
  • import jwt

📝 Açıklama:
Bu örnek, JWT tabanlı kimlik doğrulamaya sahip bir FastAPI uygulaması oluşturur. Token döndüren bir giriş endpoint’i ve kimlik doğrulama gerektiren korumalı bir rota içerir...

✅ En İyi Uygulamalar:
  • Gizli anahtarlar için environment variable kullanın
  • Production'da uygun parola hashleme uygulayın
  • Token süresi dolma ve yenileme mantığı ekleyin
  • Uygun hata yönetimi ekleyin

Sonraki örneğe geçilsin mi? (y/n): n

💾 FastAPI için öğrenme sonuçları dosyaya kaydedilsin mi? (y/n): y
   Dosya adını girin (varsayılan: fastapi_learning.json): 
   ✅ Sonuçlar fastapi_learning.json dosyasına kaydedildi

📚 Şu ana kadar öğrenilen kütüphaneler: ['FastAPI']

🔄 Başka bir kütüphane öğrenmek ister misiniz? (y/n): n

🎉 Oturum Özeti:
Başarıyla öğrenilen kütüphane sayısı: 1
  • FastAPI: 3 örnek üretildi
```


## Sonraki Adımlar

- **GitHub Entegrasyonu**: README dosyalarından ve örnek depolardan öğrenme
- **Video Eğitim İşleme**: Video dokümantasyonundan bilgi çıkarma
- **Topluluk Örnekleri**: Stack Overflow ve forumlardaki örnekleri toplama
- **Sürüm Karşılaştırması**: Kütüphane sürümleri arasındaki API değişikliklerini izleme
- **Test Üretimi**: Üretilen kodlar için otomatik birim testi oluşturma
- **Sayfa Tarama**: Kullanımı aktif olarak anlamak için dokümantasyon sayfalarını otomatik tarama

Bu eğitim, DSPy’nin bilinmeyen kütüphaneleri dokümantasyonlarından öğrenme sürecinin tamamını nasıl otomatikleştirebildiğini gösterir; bu da onu hızlı teknoloji benimseme ve keşif için değerli kılar.
