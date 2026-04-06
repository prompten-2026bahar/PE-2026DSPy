# DSPy ile Yaratıcı Bir Metin Tabanlı Yapay Zekâ Oyunu Geliştirme

Bu eğitim, DSPy’nin modüler programlama yaklaşımını kullanarak etkileşimli bir metin tabanlı macera oyununun nasıl oluşturulacağını gösterir. Yapay zekânın anlatı üretimini, karakter etkileşimlerini ve uyarlanabilir oynanışı yönettiği dinamik bir oyun geliştireceksiniz. fileciteturn33file0

## Ne İnşa Edeceksiniz

Şunları içeren akıllı bir metin tabanlı macera oyunu:

- Dinamik hikâye üretimi ve dallanan anlatılar
- Yapay zekâ destekli karakter etkileşimleri ve diyalog
- Oyuncu seçimlerine yanıt veren uyarlanabilir oynanış
- Envanter ve karakter gelişim sistemleri
- Oyun durumunu kaydetme/yükleme işlevi fileciteturn33file0

## Kurulum

```bash
pip install dspy rich typer
```

## Adım 1: Temel Oyun Çatısı

```python
import dspy
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import typer

# DSPy'yi yapılandır
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

console = Console()

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    INVENTORY = "inventory"
    CHARACTER = "character"
    GAME_OVER = "game_over"

@dataclass
class Player:
    name: str
    health: int = 100
    level: int = 1
    experience: int = 0
    inventory: list[str] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=lambda: {
        "strength": 10,
        "intelligence": 10,
        "charisma": 10,
        "stealth": 10
    })

    def add_item(self, item: str):
        self.inventory.append(item)
        console.print(f"[green]{item} envantere eklendi![/green]")

    def remove_item(self, item: str) -> bool:
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False

    def gain_experience(self, amount: int):
        self.experience += amount
        old_level = self.level
        self.level = 1 + (self.experience // 100)
        if self.level > old_level:
            console.print(f"[bold yellow]Seviye atladın! Artık seviye {self.level} oldun![/bold yellow]")

@dataclass
class GameContext:
    current_location: str = "Village Square"
    story_progress: int = 0
    visited_locations: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    game_flags: dict[str, bool] = field(default_factory=dict)

    def add_flag(self, flag: str, value: bool = True):
        self.game_flags[flag] = value

    def has_flag(self, flag: str) -> bool:
        return self.game_flags.get(flag, False)

class GameEngine:
    def __init__(self):
        self.player = None
        self.context = GameContext()
        self.state = GameState.MENU
        self.running = True

    def save_game(self, filename: str = "savegame.json"):
        """Mevcut oyun durumunu kaydet."""
        save_data = {
            "player": {
                "name": self.player.name,
                "health": self.player.health,
                "level": self.player.level,
                "experience": self.player.experience,
                "inventory": self.player.inventory,
                "skills": self.player.skills
            },
            "context": {
                "current_location": self.context.current_location,
                "story_progress": self.context.story_progress,
                "visited_locations": self.context.visited_locations,
                "npcs_met": self.context.npcs_met,
                "completed_quests": self.context.completed_quests,
                "game_flags": self.context.game_flags
            }
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        console.print(f"[green]Oyun {filename} dosyasına kaydedildi![/green]")

    def load_game(self, filename: str = "savegame.json") -> bool:
        """Oyun durumunu dosyadan yükle."""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)

            # Oyuncuyu yeniden oluştur
            player_data = save_data["player"]
            self.player = Player(
                name=player_data["name"],
                health=player_data["health"],
                level=player_data["level"],
                experience=player_data["experience"],
                inventory=player_data["inventory"],
                skills=player_data["skills"]
            )

            # Bağlamı yeniden oluştur
            context_data = save_data["context"]
            self.context = GameContext(
                current_location=context_data["current_location"],
                story_progress=context_data["story_progress"],
                visited_locations=context_data["visited_locations"],
                npcs_met=context_data["npcs_met"],
                completed_quests=context_data["completed_quests"],
                game_flags=context_data["game_flags"]
            )

            console.print(f"[green]Oyun {filename} dosyasından yüklendi![/green]")
            return True

        except FileNotFoundError:
            console.print(f"[red]Kayıt dosyası {filename} bulunamadı![/red]")
            return False
        except Exception as e:
            console.print(f"[red]Oyun yüklenirken hata oluştu: {e}![/red]")
            return False

# Oyun motorunu başlat
game = GameEngine()
```

## Adım 2: Yapay Zekâ Destekli Hikâye Üretimi

```python
class StoryGenerator(dspy.Signature):
    """Mevcut oyun durumuna göre dinamik hikâye içeriği üret."""
    location: str = dspy.InputField(desc="Mevcut konum")
    player_info: str = dspy.InputField(desc="Oyuncu bilgileri ve istatistikleri")
    story_progress: int = dspy.InputField(desc="Mevcut hikâye ilerleme seviyesi")
    recent_actions: str = dspy.InputField(desc="Oyuncunun son eylemleri")

    scene_description: str = dspy.OutputField(desc="Mevcut sahnenin canlı açıklaması")
    available_actions: list[str] = dspy.OutputField(desc="Olası oyuncu eylemlerinin listesi")
    npcs_present: list[str] = dspy.OutputField(desc="Bu konumda bulunan NPC'ler")
    items_available: list[str] = dspy.OutputField(desc="Bulunabilecek veya etkileşime geçilebilecek eşyalar")

class DialogueGenerator(dspy.Signature):
    """NPC diyalogları ve yanıtları üret."""
    npc_name: str = dspy.InputField(desc="NPC'nin adı ve türü")
    npc_personality: str = dspy.InputField(desc="NPC kişiliği ve arka planı")
    player_input: str = dspy.InputField(desc="Oyuncunun söylediği veya yaptığı şey")
    context: str = dspy.InputField(desc="Mevcut oyun bağlamı ve geçmiş")

    npc_response: str = dspy.OutputField(desc="NPC'nin diyalog yanıtı")
    mood_change: str = dspy.OutputField(desc="NPC'nin ruh hâlinin nasıl değiştiği (positive/negative/neutral)")
    quest_offered: bool = dspy.OutputField(desc="NPC'nin bir görev sunup sunmadığı")
    information_revealed: str = dspy.OutputField(desc="Paylaşılan önemli bilgiler")

class ActionResolver(dspy.Signature):
    """Oyuncu eylemlerini çözümle ve sonuçları belirle."""
    action: str = dspy.InputField(desc="Oyuncunun seçtiği eylem")
    player_stats: str = dspy.InputField(desc="Oyuncunun mevcut istatistikleri ve becerileri")
    context: str = dspy.InputField(desc="Mevcut oyun bağlamı")
    difficulty: str = dspy.InputField(desc="Eylemin zorluk seviyesi")

    success: bool = dspy.OutputField(desc="Eylemin başarılı olup olmadığı")
    outcome_description: str = dspy.OutputField(desc="Ne olduğunun açıklaması")
    stat_changes: dict[str, int] = dspy.OutputField(desc="Oyuncu istatistiklerindeki değişiklikler")
    items_gained: list[str] = dspy.OutputField(desc="Bu eylemden kazanılan eşyalar")
    experience_gained: int = dspy.OutputField(desc="Kazanılan deneyim puanları")

class GameAI(dspy.Module):
    """Oyun mantığı ve anlatısı için ana yapay zekâ modülü."""

    def __init__(self):
        super().__init__()
        self.story_gen = dspy.ChainOfThought(StoryGenerator)
        self.dialogue_gen = dspy.ChainOfThought(DialogueGenerator)
        self.action_resolver = dspy.ChainOfThought(ActionResolver)

    def generate_scene(self, player: Player, context: GameContext, recent_actions: str = "") -> Dict:
        """Mevcut sahne açıklamasını ve seçenekleri üret."""

        player_info = f"Seviye {player.level} {player.name}, Sağlık: {player.health}, Beceriler: {player.skills}"

        scene = self.story_gen(
            location=context.current_location,
            player_info=player_info,
            story_progress=context.story_progress,
            recent_actions=recent_actions
        )

        return {
            "description": scene.scene_description,
            "actions": scene.available_actions,
            "npcs": scene.npcs_present,
            "items": scene.items_available
        }

    def handle_dialogue(self, npc_name: str, player_input: str, context: GameContext) -> Dict:
        """NPC'lerle yapılan konuşmayı işle."""

        # Ad ve bağlama göre NPC kişiliği oluştur
        personality_map = {
            "Village Elder": "Bilge, bilgili, bilmece gibi konuşur, kadim bilgiye sahiptir",
            "Merchant": "Açgözlü ama adil, pazarlığı sever, değerli eşyaları bilir",
            "Guard": "Görevine bağlı, yabancılardan şüphelenir, kurallara sıkı sıkıya uyar",
            "Thief": "Sinsi, güvenilmez, gizli şeyler hakkında bilgi sahibidir",
            "Wizard": "Gizemli, güçlü, büyü ve kadim güçler hakkında konuşur"
        }

        personality = personality_map.get(npc_name, "Yerel bilgiye sahip dost canlısı köylü")
        game_context = f"Konum: {context.current_location}, Hikâye ilerlemesi: {context.story_progress}"

        response = self.dialogue_gen(
            npc_name=npc_name,
            npc_personality=personality,
            player_input=player_input,
            context=game_context
        )

        return {
            "response": response.npc_response,
            "mood": response.mood_change,
            "quest": response.quest_offered,
            "info": response.information_revealed
        }

    def resolve_action(self, action: str, player: Player, context: GameContext) -> Dict:
        """Oyuncu eylemlerini çözümle ve sonuçları belirle."""

        player_stats = f"Seviye {player.level}, Sağlık {player.health}, Beceriler: {player.skills}"
        game_context = f"Konum: {context.current_location}, İlerleme: {context.story_progress}"

        # Eylem türüne göre zorluk belirle
        difficulty = "medium"
        if any(word in action.lower() for word in ["fight", "battle", "attack"]):
            difficulty = "hard"
        elif any(word in action.lower() for word in ["look", "examine", "talk"]):
            difficulty = "easy"

        result = self.action_resolver(
            action=action,
            player_stats=player_stats,
            context=game_context,
            difficulty=difficulty
        )

        return {
            "success": result.success,
            "description": result.outcome_description,
            "stat_changes": result.stat_changes,
            "items": result.items_gained,
            "experience": result.experience_gained
        }

# Yapay zekâyı başlat
ai = GameAI()
```

## Adım 3: Oyun Arayüzü ve Etkileşim

```python
def display_game_header():
    """Oyun başlığını göster."""
    header = Text("🏰 MYSTIC REALM ADVENTURE 🏰", style="bold magenta")
    console.print(Panel(header, style="bright_blue"))

def display_player_status(player: Player):
    """Oyuncu durum panelini göster."""
    status = f"""
[bold]Ad:[/bold] {player.name}
[bold]Seviye:[/bold] {player.level} (XP: {player.experience})
[bold]Sağlık:[/bold] {player.health}/100
[bold]Beceriler:[/bold]
  • Güç: {player.skills['strength']}
  • Zekâ: {player.skills['intelligence']}
  • Karizma: {player.skills['charisma']}
  • Gizlilik: {player.skills['stealth']}
[bold]Envanter:[/bold] {len(player.inventory)} eşya
    """
    console.print(Panel(status.strip(), title="Oyuncu Durumu", style="green"))

def display_location(context: GameContext, scene: Dict):
    """Mevcut konumu ve sahneyi göster."""
    location_panel = f"""
[bold yellow]{context.current_location}[/bold yellow]

{scene['description']}
    """

    if scene['npcs']:
        location_panel += f"\n\n[bold]Bulunan NPC'ler:[/bold] {', '.join(scene['npcs'])}"

    if scene['items']:
        location_panel += f"\n[bold]Görünen eşyalar:[/bold] {', '.join(scene['items'])}"

    console.print(Panel(location_panel.strip(), title="Mevcut Konum", style="cyan"))

def display_actions(actions: list[str]):
    """Mevcut eylemleri göster."""
    action_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
    console.print(Panel(action_text, title="Mevcut Eylemler", style="yellow"))

def get_player_choice(max_choices: int) -> int:
    """Girdi doğrulamasıyla oyuncu seçimini al."""
    while True:
        try:
            choice = typer.prompt("Bir eylem seç (numara)")
            choice_num = int(choice)
            if 1 <= choice_num <= max_choices:
                return choice_num - 1
            else:
                console.print(f"[red]Lütfen 1 ile {max_choices} arasında bir sayı girin[/red]")
        except ValueError:
            console.print("[red]Lütfen geçerli bir sayı girin[/red]")

def show_inventory(player: Player):
    """Oyuncu envanterini göster."""
    if not player.inventory:
        console.print(Panel("Envanterin boş.", title="Envanter", style="red"))
    else:
        items = "\n".join([f"• {item}" for item in player.inventory])
        console.print(Panel(items, title="Envanter", style="green"))

def main_menu():
    """Ana menüyü göster ve seçimi işle."""
    console.clear()
    display_game_header()

    menu_options = [
        "1. Yeni Oyun",
        "2. Oyun Yükle", 
        "3. Nasıl Oynanır",
        "4. Çıkış"
    ]

    menu_text = "\n".join(menu_options)
    console.print(Panel(menu_text, title="Ana Menü", style="bright_blue"))

    choice = typer.prompt("Bir seçenek seç")
    return choice

def show_help():
    """Yardım bilgisini göster."""
    help_text = """
[bold]Nasıl Oynanır:[/bold]

• Bu, yapay zekâ destekli metin tabanlı bir macera oyunudur
• Numaralı seçenekleri seçerek karar verirsiniz
• Dünya hakkında bilgi edinmek ve görev almak için NPC'lerle konuşun
• Eşya ve macera bulmak için farklı konumları keşfedin
• Seçimleriniz hikâyeyi ve karakter gelişimini etkiler
• Eşyalarınızı görmek için 'inventory' kullanın
• Karakter bilgilerinizi görmek için 'status' kullanın
• İlerlemenizi kaydetmek için 'save' yazın
• Ana menüye dönmek için 'quit' yazın

[bold]İpuçları:[/bold]
• Farklı beceriler çeşitli eylemlerde başarınızı etkiler
• NPC'ler geçmiş etkileşimlerinizi hatırlar
• İyice keşfedin - gizli sırlar var!
• Ününüz, NPC'lerin size nasıl davrandığını etkiler
    """
    console.print(Panel(help_text.strip(), title="Oyun Yardımı", style="blue"))
    typer.prompt("Devam etmek için Enter'a bas")
```

## Adım 4: Ana Oyun Döngüsü

```python
def create_new_character():
    """Yeni bir oyuncu karakteri oluştur."""
    console.clear()
    display_game_header()

    name = typer.prompt("Karakterinin adını gir")

    # Beceri puanı dağıtımı ile karakter oluşturma
    console.print("\n[bold]Karakter Oluşturma[/bold]")
    console.print("Becerilerin arasında dağıtmak için 10 ek beceri puanına sahipsin.")
    console.print("Temel beceriler 10 puanla başlar.\n")

    skills = {"strength": 10, "intelligence": 10, "charisma": 10, "stealth": 10}
    points_remaining = 10

    for skill in skills.keys():
        if points_remaining > 0:
            console.print(f"Kalan puan: {points_remaining}")
            while True:
                try:
                    points = int(typer.prompt(f"{skill} için eklenecek puan (0-{points_remaining})"))
                    if 0 <= points <= points_remaining:
                        skills[skill] += points
                        points_remaining -= points
                        break
                    else:
                        console.print(f"[red]0 ile {points_remaining} arasında bir sayı girin[/red]")
                except ValueError:
                    console.print("[red]Lütfen geçerli bir sayı girin[/red]")

    player = Player(name=name, skills=skills)
    console.print(f"\n[green]Mystic Realm'e hoş geldin, {name}![/green]")
    return player

def game_loop():
    """Ana oyun döngüsü."""
    recent_actions = ""

    while game.running and game.state == GameState.PLAYING:
        console.clear()
        display_game_header()

        # Mevcut sahneyi üret
        scene = ai.generate_scene(game.player, game.context, recent_actions)

        # Oyun durumunu göster
        display_player_status(game.player)
        display_location(game.context, scene)

        # Standart eylemleri ekle
        all_actions = scene['actions'] + ["Envanteri kontrol et", "Karakter durumu", "Oyunu kaydet", "Menüye dön"]
        display_actions(all_actions)

        # Oyuncu seçimini al
        choice_idx = get_player_choice(len(all_actions))
        chosen_action = all_actions[choice_idx]

        # Özel komutları işle
        if chosen_action == "Envanteri kontrol et":
            show_inventory(game.player)
            typer.prompt("Devam etmek için Enter'a bas")
            continue
        elif chosen_action == "Karakter durumu":
            display_player_status(game.player)
            typer.prompt("Devam etmek için Enter'a bas")
            continue
        elif chosen_action == "Oyunu kaydet":
            game.save_game()
            typer.prompt("Devam etmek için Enter'a bas")
            continue
        elif chosen_action == "Menüye dön":
            game.state = GameState.MENU
            break

        # Oyun eylemlerini işle
        if chosen_action in scene['actions']:
            # Bunun bir NPC ile diyalog olup olmadığını kontrol et
            npc_target = None
            for npc in scene['npcs']:
                if npc.lower() in chosen_action.lower():
                    npc_target = npc
                    break

            if npc_target:
                # NPC etkileşimini işle
                console.print(f"\n[bold]{npc_target} ile konuşuluyor...[/bold]")
                dialogue = ai.handle_dialogue(npc_target, chosen_action, game.context)

                console.print(f"\n[italic]{npc_target}:[/italic] \"{dialogue['response']}\"")

                if dialogue['quest']:
                    console.print(f"[yellow]💼 Görev fırsatı tespit edildi![/yellow]")

                if dialogue['info']:
                    console.print(f"[blue]ℹ️  {dialogue['info']}[/blue]")

                # NPC'yi tanışılanlar listesine ekle
                if npc_target not in game.context.npcs_met:
                    game.context.npcs_met.append(npc_target)

                recent_actions = f"{npc_target} ile konuşuldu: {chosen_action}"
            else:
                # Genel eylemi işle
                result = ai.resolve_action(chosen_action, game.player, game.context)

                console.print(f"\n{result['description']}")

                # Sonuçları uygula
                if result['success']:
                    console.print("[green]✅ Başarılı![/green]")

                    # İstatistik değişikliklerini uygula
                    for stat, change in result['stat_changes'].items():
                        if stat in game.player.skills:
                            game.player.skills[stat] += change
                            if change > 0:
                                console.print(f"[green]{stat.title()} {change} arttı![/green]")
                        elif stat == "health":
                            game.player.health = max(0, min(100, game.player.health + change))
                            if change > 0:
                                console.print(f"[green]Sağlık {change} arttı![/green]")
                            elif change < 0:
                                console.print(f"[red]Sağlık {abs(change)} azaldı![/red]")

                    # Eşyaları ekle
                    for item in result['items']:
                        game.player.add_item(item)

                    # Deneyim ver
                    if result['experience'] > 0:
                        game.player.gain_experience(result['experience'])

                    # Hikâye ilerlemesini güncelle
                    game.context.story_progress += 1
                else:
                    console.print("[red]❌ Eylem planlandığı gibi gitmedi...[/red]")

                recent_actions = f"Denenen eylem: {chosen_action}"

            # Oyun bitiş koşullarını kontrol et
            if game.player.health <= 0:
                console.print("\n[bold red]💀 Öldün! Oyun Bitti![/bold red]")
                game.state = GameState.GAME_OVER
                break

            typer.prompt("\nDevam etmek için Enter'a bas")

def main():
    """Ana oyun fonksiyonu."""
    while game.running:
        if game.state == GameState.MENU:
            choice = main_menu()

            if choice == "1":
                game.player = create_new_character()
                game.context = GameContext()
                game.state = GameState.PLAYING
                console.print("\n[italic]Maceran başlıyor...[/italic]")
                typer.prompt("Başlamak için Enter'a bas")

            elif choice == "2":
                if game.load_game():
                    game.state = GameState.PLAYING
                typer.prompt("Devam etmek için Enter'a bas")

            elif choice == "3":
                show_help()

            elif choice == "4":
                game.running = False
                console.print("[bold]Oynadığın için teşekkürler! Hoşça kal![/bold]")

        elif game.state == GameState.PLAYING:
            game_loop()

        elif game.state == GameState.GAME_OVER:
            console.print("\n[bold]Oyun Bitti[/bold]")
            restart = typer.confirm("Ana menüye dönmek ister misin?")
            if restart:
                game.state = GameState.MENU
            else:
                game.running = False

if __name__ == "__main__":
    main()
```

## Örnek Oynanış

Oyunu çalıştırdığınızda şunları deneyimlersiniz:

**Karakter Oluşturma:**
```
🏰 MYSTIC REALM ADVENTURE 🏰

Karakterinin adını gir: Aria

Karakter Oluşturma
Becerilerin arasında dağıtmak için 10 ek beceri puanına sahipsin.
Temel beceriler 10 puanla başlar.

Kalan puan: 10
strength için eklenecek puan (0-10): 2
intelligence için eklenecek puan (0-8): 4
charisma için eklenecek puan (0-4): 3
stealth için eklenecek puan (0-1): 1

Mystic Realm'e hoş geldin, Aria!
```

**Dinamik Sahne Üretimi:**
```
┌──────────── Current Location ────────────┐
│ Village Square                           │
│                                          │
│ You stand in the bustling heart of       │
│ Willowbrook Village. The ancient stone   │
│ fountain bubbles cheerfully as merchants │
│ hawk their wares and children play. A    │
│ mysterious hooded figure lurks near the  │
│ shadows of the old oak tree.             │
│                                          │
│ NPCs present: Village Elder, Merchant    │
│ Items visible: Strange Medallion, Herbs  │
└──────────────────────────────────────────┘

┌────────── Available Actions ─────────────┐
│ 1. Approach the hooded figure            │
│ 2. Talk to the Village Elder             │
│ 3. Browse the merchant's wares           │
│ 4. Examine the strange medallion         │
│ 5. Gather herbs near the fountain        │
│ 6. Head to the forest path               │
└───────────────────────────────────────────┘
```

**Yapay Zekâ ile Üretilen Diyalog:**
```
Talking to Village Elder...

Village Elder: "Ah, young traveler, I sense a great destiny 
surrounds you like morning mist. The ancient prophecy speaks 
of one who would come bearing the mark of courage. Tell me, 
have you noticed anything... unusual in your travels?"

💼 Quest opportunity detected!
ℹ️ The Village Elder knows about an ancient prophecy that might involve you
```

## Sonraki Adımlar

- **Savaş Sistemi**: Strateji içeren sıra tabanlı savaşlar ekleyin
- **Büyü Sistemi**: Kaynak yönetimli büyü kullanımı ekleyin
- **Çok Oyunculu**: İşbirlikçi maceralar için ağ desteği ekleyin
- **Görev Sistemi**: Dallanan sonuçlara sahip karmaşık çok adımlı görevler oluşturun
- **Dünya İnşası**: Prosedürel olarak üretilen konumlar ve karakterler ekleyin
- **Ses**: Ses efektleri ve arka plan müziği ekleyin

Bu eğitim, DSPy’nin modüler yaklaşımının, yapay zekânın yaratıcı içerik üretimini üstlenirken aynı zamanda tutarlı oyun mantığını ve oyuncu iradesini koruyan karmaşık, etkileşimli sistemleri nasıl mümkün kıldığını gösterir.
