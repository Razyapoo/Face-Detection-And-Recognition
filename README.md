Real time Face Detection and Recognition
===
Cílem je vyvinout efektivní metody rozpoznání obličejů v reálném čase pro odemčení dveří. 
Implementaci tedy můžeme shrnout do následujících fází:
 • Detekce obličeje
 • Natrénování dat 
 • Rozpoznání obličeje 
 
Aplikace je napsána v jazyce C++ s využitím knihovny počítačového vidění OpenCV.

# 
***Jak funguje celý proces detekce a rozpoznávání tváře?***

Nejprve musíme načíst vstupní obrázek z kamery. Jako vstupní zařízení použijeme webovoukamerupočítače.Důležité je správně nastavit parametry při získávání přístupu ke kameře. 

Před tím než začneme samotné rozpoznání testovacích dat,musíme správně načíst a zpracovat trénovací sadu. V průběhu zpracovaní trénovací sady extrahujeme z každého obrázku z databáze jednotlivé příznaky. Takovým způsobem se vytvoří (natrénuje) model, který pak použijeme pro klasifikaci testovacích dat. Je to téměř nejdůležitější část celého procesu, které je nutné věnovat velkou pozornost. Důležité je správně zpracovat obrázek (např. zarovnat podle očí, redukovat šum atd.), jelikož na tom záleží přesnost rozpoznání osoby.

 Pokud natrénování modelu proběhlo úspěšně, začneme proces klasifikace testovacích dat. Jelikož jíž máme načtený vstupní obrázek, převedeme ho na stupně šedi a detekujeme oblast obsahující obličej. Zpracujeme testovací obrázek a extrahujeme z něj příznaky stejným způsobem jako z trénovacích dat, jelikož pro porovnání potřebujeme data ve stejné kondici. 
 
 Jako poslední fáze procesu klasifikujeme vstupní testovací obrázek na základě nejkratší vzdálenosti mezi vyextrahovanými příznaky tohoto obrázku a natrénovaných dat. 
 
 Všechny fáze procesu detekce a rozpoznání tváře jsou důležité, protože jsou propojené mezi sebou a každá následující fáze je závislá na předchozí.
