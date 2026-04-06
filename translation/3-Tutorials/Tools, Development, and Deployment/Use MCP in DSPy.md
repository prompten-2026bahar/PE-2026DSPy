# Eğitim: DSPy’de MCP araçlarını kullanma

MCP, açılımı Model Context Protocol olan, uygulamaların LLM’lere nasıl bağlam sağladığını standartlaştıran açık bir protokoldür. Bir miktar geliştirme yükü getirmesine rağmen MCP, kullandığınız teknik yığından bağımsız olarak diğer geliştiricilerle araçları, kaynakları ve istemleri paylaşmak için değerli bir fırsat sunar. Aynı şekilde, diğer geliştiriciler tarafından oluşturulan araçları kodu yeniden yazmadan kullanabilirsiniz.

Bu kılavuzda, DSPy içinde MCP araçlarının nasıl kullanılacağını adım adım inceleyeceğiz. Gösterim amacıyla, kullanıcıların uçuş rezervasyonu yapmasına ve mevcut rezervasyonlarını değiştirmesine veya iptal etmesine yardımcı olabilecek bir havayolu hizmet ajanı oluşturacağız. Bu, özel araçlara sahip bir MCP sunucusuna dayanacaktır; ancak bunu [topluluk tarafından oluşturulan MCP sunucularına](https://modelcontextprotocol.io/examples) genellemek kolay olmalıdır.

??? "Bu eğitim nasıl çalıştırılır"
    Bu eğitim, Google Colab veya Databricks not defterleri gibi barındırılan IPython not defterlerinde çalıştırılamaz.
    Kodu çalıştırmak için, yerel cihazınızda kod yazma kılavuzunu izlemeniz gerekir. Kod macOS üzerinde test edilmiştir ve Linux ortamlarında da aynı şekilde çalışmalıdır.

## Bağımlılıkları Yükleme

Başlamadan önce gerekli bağımlılıkları yükleyelim:

```shell
pip install -U "dspy[mcp]"
```

## MCP Sunucusu Kurulumu

Önce havayolu ajanı için MCP sunucusunu kuralım. Bu sunucu şunları içerir:

- Bir dizi veritabanı
  - Kullanıcı bilgilerini saklayan kullanıcı veritabanı.
  - Uçuş bilgilerini saklayan uçuş veritabanı.
  - Müşteri biletlerini saklayan bilet veritabanı.
- Bir dizi araç
  - fetch_flight_info: belirli tarihler için uçuş bilgilerini alır.
  - fetch_itinerary: rezerve edilmiş seyahat planlarının bilgilerini alır.
  - book_itinerary: kullanıcı adına uçuş rezervasyonu yapar.
  - modify_itinerary: uçuş değişikliği veya iptal yoluyla bir seyahat planını değiştirir.
  - get_user_info: kullanıcı bilgilerini alır.
  - file_ticket: insan desteği için bir bekleyen iş bileti oluşturur.

Çalışma dizininizde `mcp_server.py` adlı bir dosya oluşturun ve aşağıdaki içeriği içine yapıştırın:

```python
import random
import string

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Bir MCP sunucusu oluştur
mcp = FastMCP("Airline Agent")


class Date(BaseModel):
    # Nedense LLM, `datetime.datetime` belirtmekte başarısız oluyor
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}


@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Verilen tarihte başlangıç noktasından varış noktasına uçuş bilgilerini getir"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights


@mcp.tool()
def fetch_itinerary(confirmation_number: str):
    """Veritabanından rezerve edilmiş bir seyahat planı bilgisini getir"""
    return itinery_database.get(confirmation_number)


@mcp.tool()
def pick_flight(flights: list[Flight]):
    """Kullanıcının isteğine en uygun uçuşu seç."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]


def generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


@mcp.tool()
def book_itinerary(flight: Flight, user_profile: UserProfile):
    """Kullanıcı adına uçuş rezervasyonu yap."""
    confirmation_number = generate_id()
    while confirmation_number in itinery_database:
        confirmation_number = generate_id()
    itinery_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinery_database[confirmation_number]


@mcp.tool()
def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Kullanıcı adına bir seyahat planını iptal et."""
    if confirmation_number in itinery_database:
        del itinery_database[confirmation_number]
        return
    raise ValueError("Seyahat planı bulunamadı, lütfen onay numaranızı kontrol edin.")


@mcp.tool()
def get_user_info(name: str):
    """Verilen adla veritabanından kullanıcı profilini getir."""
    return user_database.get(name)


@mcp.tool()
def file_ticket(user_request: str, user_profile: UserProfile):
    """Ajanın çözemediği durumlarda bir müşteri destek bileti oluştur."""
    ticket_id = generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id


if __name__ == "__main__":
    mcp.run()
```

Sunucuyu başlatmadan önce koda hızlıca bakalım.

Önce bir `FastMCP` örneği oluşturuyoruz; bu, hızlıca bir MCP sunucusu kurmaya yardımcı olan bir yardımcı araçtır:

```python
mcp = FastMCP("Airline Agent")
```

Ardından veri yapılarımızı tanımlıyoruz; bunlar gerçek bir uygulamada veritabanı şeması olurdu, örneğin:

```python
class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float
```

Bunun ardından veritabanı örneklerimizi başlatıyoruz. Gerçek bir uygulamada bunlar gerçek veritabanlarına bağlanan konektörler olurdu; ancak basitlik için yalnızca sözlükler kullanıyoruz:

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}
```

Sonraki adım, araçları tanımlayıp `@mcp.tool()` ile işaretlemektir; böylece MCP istemcileri tarafından MCP araçları olarak keşfedilebilirler:

```python
@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Verilen tarihte başlangıç noktasından varış noktasına uçuş bilgilerini getir"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights
```

Son adım ise sunucuyu ayağa kaldırmaktır:

```python
if __name__ == "__main__":
    mcp.run()
```

Artık sunucuyu yazmayı bitirdik. Şimdi başlatalım:

```shell
python path_to_your_working_directory/mcp_server.py
```

## MCP Sunucusundaki Araçları Kullanan Bir DSPy Programı Yazma

Sunucu artık çalıştığına göre, kullanıcıya yardımcı olmak için sunucumuzdaki MCP araçlarını kullanan gerçek havayolu hizmet ajanını oluşturalım. Çalışma dizininizde `dspy_mcp_agent.py` adlı bir dosya oluşturun ve içine kod eklemek için kılavuzu izleyin.

### MCP Sunucularından Araçları Toplama

Öncelikle MCP sunucusundan mevcut tüm araçları toplamamız ve bunları DSPy tarafından kullanılabilir hale getirmemiz gerekir. DSPy, standart araç arayüzü olarak [`dspy.Tool`](https://dspy.ai/api/primitives/Tool/) API’sini sağlar. Tüm MCP araçlarını `dspy.Tool` nesnelerine dönüştürelim.

MCP sunucusuyla iletişim kurmak, mevcut araçları çekmek ve bunları `from_mcp_tool` statik metodu ile `dspy.Tool` nesnelerine dönüştürmek için bir MCP istemci örneği oluşturmamız gerekir:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# stdio bağlantısı için sunucu parametrelerini oluştur
server_params = StdioServerParameters(
    command="python",  # Çalıştırılabilir dosya
    args=["path_to_your_working_directory/mcp_server.py"],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Bağlantıyı başlat
            await session.initialize()
            # Kullanılabilir araçları listele
            tools = await session.list_tools()

            # MCP araçlarını DSPy araçlarına dönüştür
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

Yukarıdaki kodla, mevcut tüm MCP araçlarını başarıyla toplayıp DSPy araçlarına dönüştürmüş olduk.

### Müşteri İsteklerini Ele Almak İçin Bir DSPy Ajanı Oluşturma

Şimdi müşteri isteklerini ele almak için ajanı oluşturmak üzere `dspy.ReAct` kullanacağız. `ReAct`, “reasoning and acting” anlamına gelir; yani LLM’den bir aracı çağırıp çağırmayacağına veya süreci sonlandırıp sonlandırmayacağına karar vermesi istenir. Bir araç gerekiyorsa, hangi aracın çağrılacağına ve uygun argümanların ne olacağına karar verme sorumluluğu LLM’dedir.

Her zamanki gibi, ajanımızın giriş ve çıkışlarını tanımlamak için bir `dspy.Signature` oluşturmamız gerekir:

```python
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """Sen bir havayolu müşteri hizmetleri ajanısın. Kullanıcı isteklerini ele almak için sana bir araç listesi verilir. Kullanıcıların isteklerini yerine getirmek için doğru aracı kullanmaya karar vermelisin."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Süreç sonucunu ve kullanıcıların ihtiyaç duyduğu bilgileri özetleyen mesaj, "
            "örneğin bu bir uçuş rezervasyonu isteğiyse confirmation_number."
        )
    )
```

Ve ajanımız için bir LM seçelim:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

Ardından araçları ve imzayı `dspy.ReAct` API’sine vererek ReAct ajanını oluşturuyoruz. Artık tam kod betiğini bir araya getirebiliriz:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy

# stdio bağlantısı için sunucu parametrelerini oluştur
server_params = StdioServerParameters(
    command="python",  # Çalıştırılabilir dosya
    args=["script_tmp/mcp_server.py"],  # İsteğe bağlı komut satırı argümanları
    env=None,  # İsteğe bağlı ortam değişkenleri
)


class DSPyAirlineCustomerService(dspy.Signature):
    """Sen bir havayolu müşteri hizmetleri ajanısın. Kullanıcı isteklerini ele almak için sana bir araç listesi verilir.
    Kullanıcıların isteklerini yerine getirmek için doğru aracı kullanmaya karar vermelisin."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Süreç sonucunu ve kullanıcıların ihtiyaç duyduğu bilgileri özetleyen mesaj, "
            "örneğin bu bir uçuş rezervasyonu isteğiyse confirmation_number."
        )
    )


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Bağlantıyı başlat
            await session.initialize()
            # Kullanılabilir araçları listele
            tools = await session.list_tools()

            # MCP araçlarını DSPy araçlarına dönüştür
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # Ajanı oluştur
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run("lütfen 09/01/2025 tarihinde SFO'dan JFK'ye bir uçuş rezervasyonu yapmama yardım et, adım Adam"))
```

MCP araçları varsayılan olarak async olduğu için `react.acall` çağırmamız gerektiğine dikkat edin. Şimdi betiği çalıştıralım:

```shell
python path_to_your_working_directory/dspy_mcp_agent.py
```

Aşağıdakine benzer bir çıktı görmelisiniz:

```
Prediction(
    trajectory={'thought_0': 'I need to fetch flight information for Adam from SFO to JFK on 09/01/2025 to find available flights for booking.', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': ['{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}', '{"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}'], ..., 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},
    reasoning="I successfully booked a flight for Adam from SFO to JFK on 09/01/2025. I found two available flights, selected the more economical option (flight DA123 at 1 AM for $200), retrieved Adam's user profile, and completed the booking process. The confirmation number for the flight is 8h7clk3q.",
    process_result='Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 8h7clk3q.'
)
```

`trajectory` alanı, tüm düşünme ve eylem sürecini içerir. Perde arkasında neler olduğuna meraklıysan, `dspy.ReAct` içinde gerçekleşen her adımı görselleştiren MLflow kurulumunu yapmak için [Observability Guide](https://dspy.ai/tutorials/observability/) kılavuzuna göz at.

## Sonuç

Bu kılavuzda, özel bir MCP sunucusu ve `dspy.ReAct` modülünü kullanan bir havayolu hizmet ajanı oluşturduk. MCP desteği bağlamında DSPy, MCP araçlarıyla etkileşim kurmak için basit bir arayüz sağlar ve ihtiyaç duyduğunuz her türlü işlevi uygulama esnekliği verir.
