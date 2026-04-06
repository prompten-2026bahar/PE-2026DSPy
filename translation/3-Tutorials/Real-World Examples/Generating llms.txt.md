# DSPy ile Kod Dokümantasyonu için llms.txt Üretme

Bu eğitim, DSPy deposunun kendisi için otomatik olarak bir `llms.txt` dosyası üretmek amacıyla DSPy’nin nasıl kullanılacağını gösterir. `llms.txt` standardı, yapay zekâ sistemlerinin kod tabanlarını daha iyi anlamasına yardımcı olan, LLM dostu dokümantasyon sağlar.

## llms.txt Nedir?

`llms.txt`, bir proje hakkında yapılandırılmış, LLM dostu dokümantasyon sağlamaya yönelik önerilen bir standarttır. Genellikle şunları içerir:

- Projeye genel bakış ve amacı
- Temel kavramlar ve terminoloji
- Mimari ve yapı
- Kullanım örnekleri
- Önemli dosya ve dizinler

## llms.txt Üretimi için Bir DSPy Programı Oluşturma

Bir depoyu analiz eden ve kapsamlı `llms.txt` dokümantasyonu üreten bir DSPy programı oluşturalım.

### Adım 1: İmzalarımızı Tanımlama

Önce, dokümantasyon üretiminin farklı yönleri için imzalar tanımlayacağız:

```python
import dspy
from typing import List

class AnalyzeRepository(dspy.Signature):
    """Bir depo yapısını analiz et ve temel bileşenleri belirle."""
    repo_url: str = dspy.InputField(desc="GitHub depo URL'si")
    file_tree: str = dspy.InputField(desc="Depo dosya yapısı")
    readme_content: str = dspy.InputField(desc="README.md içeriği")

    project_purpose: str = dspy.OutputField(desc="Projenin ana amacı ve hedefleri")
    key_concepts: list[str] = dspy.OutputField(desc="Önemli kavramlar ve terminolojiden oluşan liste")
    architecture_overview: str = dspy.OutputField(desc="Yüksek seviyeli mimari açıklaması")

class AnalyzeCodeStructure(dspy.Signature):
    """Önemli dizinleri ve dosyaları belirlemek için kod yapısını analiz et."""
    file_tree: str = dspy.InputField(desc="Depo dosya yapısı")
    package_files: str = dspy.InputField(desc="Temel paket ve yapılandırma dosyaları")

    important_directories: list[str] = dspy.OutputField(desc="Temel dizinler ve amaçları")
    entry_points: list[str] = dspy.OutputField(desc="Ana giriş noktaları ve önemli dosyalar")
    development_info: str = dspy.OutputField(desc="Geliştirme kurulumu ve iş akışı bilgisi")

class GenerateLLMsTxt(dspy.Signature):
    """Analiz edilmiş depo bilgisinden kapsamlı bir llms.txt dosyası üret."""
    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="Yaygın kullanım kalıpları ve örnekleri")

    llms_txt_content: str = dspy.OutputField(desc="Standart biçimi izleyen eksiksiz llms.txt dosya içeriği")
```

### Adım 2: Depo Analiz Edici Modülünü Oluşturma

```python
class RepositoryAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepository)
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)
        self.generate_examples = dspy.ChainOfThought("repo_info -> usage_examples")
        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)

    def forward(self, repo_url, file_tree, readme_content, package_files):
        # Depo amacı ve kavramlarını analiz et
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content
        )

        # Kod yapısını analiz et
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files
        )

        # Kullanım örnekleri üret
        usage_examples = self.generate_examples(
            repo_info=f"Amaç: {repo_analysis.project_purpose}\nKavramlar: {repo_analysis.key_concepts}"
        )

        # Nihai llms.txt dosyasını üret
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples
        )

        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis
        )
```

### Adım 3: Depo Bilgilerini Toplama

Depo bilgilerini çıkarmak için yardımcı fonksiyonlar oluşturalım:

```python
import requests
import os
from pathlib import Path

os.environ["GITHUB_ACCESS_TOKEN"] = "<your_access_token>"

def get_github_file_tree(repo_url):
    """GitHub API'sinden depo dosya yapısını al."""
    # URL'den owner/repo bilgisini çıkar
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })

    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"Depo ağacı alınamadı: {response.status_code}")

def get_github_file_content(repo_url, file_path):
    """GitHub'dan belirli bir dosyanın içeriğini al."""
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })

    if response.status_code == 200:
        import base64
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content
    else:
        return f"{file_path} alınamadı"

def gather_repository_info(repo_url):
    """Gerekli tüm depo bilgilerini topla."""
    file_tree = get_github_file_tree(repo_url)
    readme_content = get_github_file_content(repo_url, "README.md")

    # Temel paket dosyalarını al
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json"]:
        try:
            content = get_github_file_content(repo_url, file_path)
            if "Could not fetch" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except:
            continue

    package_files_content = "\n\n".join(package_files)

    return file_tree, readme_content, package_files_content
```

### Adım 4: DSPy’yi Yapılandırma ve llms.txt Üretme

```python
def generate_llms_txt_for_dspy():
    # DSPy'yi yapılandırın (tercih ettiğiniz LM'i kullanın)
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"

    # Analiz edicimizi başlat
    analyzer = RepositoryAnalyzer()

    # DSPy depo bilgilerini topla
    repo_url = "https://github.com/stanfordnlp/dspy"
    file_tree, readme_content, package_files = gather_repository_info(repo_url)

    # llms.txt üret
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files
    )

    return result

# Üretimi çalıştır
if __name__ == "__main__":
    result = generate_llms_txt_for_dspy()

    # Üretilen llms.txt dosyasını kaydet
    with open("llms.txt", "w") as f:
        f.write(result.llms_txt_content)

    print("llms.txt dosyası üretildi!")
    print("\nÖnizleme:")
    print(result.llms_txt_content[:500] + "...")
```

## Beklenen Çıktı Yapısı

DSPy için üretilen `llms.txt` aşağıdaki yapıyı izler:

```
# DSPy: Programming Language Models

## Project Overview
DSPy is a framework for programming—rather than prompting—language models...

## Key Concepts
- **Modules**: Building blocks for LM programs
- **Signatures**: Input/output specifications  
- **Teleprompters**: Optimization algorithms
- **Predictors**: Core reasoning components

## Architecture
- `/dspy/`: Main package directory
  - `/adapters/`: Input/output format handlers
  - `/clients/`: LM client interfaces
  - `/predict/`: Core prediction modules
  - `/teleprompt/`: Optimization algorithms

## Usage Examples
1. **Building a Classifier**: Using DSPy, a user can define a modular classifier that takes in text data and categorizes it into predefined classes. The user can specify the classification logic declaratively, allowing for easy adjustments and optimizations.
2. **Creating a RAG Pipeline**: A developer can implement a retrieval-augmented generation pipeline that first retrieves relevant documents based on a query and then generates a coherent response using those documents. DSPy facilitates the integration of retrieval and generation components seamlessly.
3. **Optimizing Prompts**: Users can leverage DSPy to create a system that automatically optimizes prompts for language models based on performance metrics, improving the quality of responses over time without manual intervention.
4. **Implementing Agent Loops**: A user can design an agent loop that continuously interacts with users, learns from feedback, and refines its responses, showcasing the self-improving capabilities of the DSPy framework.
5. **Compositional Code**: Developers can write compositional code that allows different modules of the AI system to interact with each other, enabling complex workflows that can be easily modified and extended.
```

Ortaya çıkan `llms.txt` dosyası, diğer yapay zekâ sistemlerinin kod tabanını daha iyi anlamasına ve onunla daha verimli çalışmasına yardımcı olabilecek, DSPy deposuna ilişkin kapsamlı ve LLM dostu bir genel bakış sunar.

## Sonraki Adımlar

- Programı birden fazla depoyu analiz edecek şekilde genişletin
- Farklı dokümantasyon biçimleri için destek ekleyin
- Dokümantasyon kalitesini değerlendirmek için metrikler oluşturun
- Etkileşimli depo analizi için bir web arayüzü geliştirin
