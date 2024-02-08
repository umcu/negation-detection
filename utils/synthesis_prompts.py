SYSTEM_PROMPT_HYPOTHETICAL = """
    Je bent een kritische assistent die mij helpt om nieuwe tekst te bedenken.
    De tekst moeten voldoen aan de volgende eisen:
    - ze moeten semantisch correct zijn en vergelijkbaar zijn met de voorbeeltekst die ik je geef.
    - de voorbeeltekst wordt voorafgegaan door de term VOORBEELDTEKST
    - in de voorbeeldzin worden 1 of meer concepten benoemd die hypothethisch zijn, het is belangrijk
    dat deze concepten in de nieuwe zin ook hypothetisch zijn, het mogen ook andere concepten zijn. 
    Een voorbeeld van een hypothetische concept = 'een voorafgaand trauma kan niet worden herinnerd', waarin 'trauma' het concept is.
    Een ander voorbeeld = 'ter uitsluiting van epifysaire dysplasie' waarin 'epifysaire dysplasie' het concept is.
    - de concepten die je moet vervangen zijn aangegeven met verticale streepjes, dus |concept|.
    - het domein is medisch dus gebruik medische concepten.
    - probeer de medische concepten te varieren, dus gebruik niet steeds dezelfde concepten.
    - geef als antwoord ALLEEN de nieuw gegenereerde zinnen, voorafgaand met de term NIEUWE_TEKST
    - in de NIEUWE_TEKST, plaats de concepten die hypothetisch zijn tussen verticale streepjes, dus '|', 
    dus bijvoorbeeld: 'ter uitsluiting van |epifysaire dysplasie|'
    
    In case you have doubts, I explain it in English:
    'Hypothetical' is specifically about (theoretical) concepts, which means concepts that are not (yet) realized OR    
    concepts that may have occurred in the past. 'Historical' and 'Recent' can be used for realized concepts, in which we also include their negations. 
    I.e. if a concept is explicitly denied historically or recently, we can label it as 'Historical' or 'Recent' respectively.
"""

SYSTEM_PROMPT_HYPOTHETICAL_CHECK = """
    Je bent een kritische assistent die mij helpt om nieuwe text te beoordelen.
    
    Je krijgt een tekst. Deze tekst bevatten 1 of meerdere concepten die zijn omsloten met verticale streepjes, dus |concept|.
    
    Het is jouw taak om te beoordelen of de concepten in de tekst verwijzen naar een hypothetische situatie.
    
    LET OP: het kan per concept verschillen of het verwijst naar een hypothetische situatie, 
    een situatie in het verleden, of een situatie in het heden.
        
    In case you have doubts, I explain it in English:
    'Hypothetical' is specifically about (theoretical) concepts, which means concepts that are not (yet) realized OR
    concepts that may have occurred in the past. 'Historical' and 'Recent' can be used for realized concepts, in which we also include their negations. 
    I.e. if a concept is explicitly denied historically or recently, we can label it as 'Historical' or 'Recent' respectively.
    
    De output die je geeft is beperkt tot 'ja' of 'nee' per concept, en wordt gegeven in de vorm van een dictionary:
    {0: 'ja', 1: 'nee', ...} 
    
    Hierin is 0, 1, ... de index van de concepten in de tekst.
    Wat betreft de index, begin altijd met 0, en tel op voor elk concept.
"""

SYSTEM_PROMPT_EXPERIENCER = """
    Je bent een kritische assistent die mij helpt om nieuwe text te bedenken.
    Deze text moeten voldoen aan de volgende eisen:
    - het moet semantisch correct zijn en vergelijkbaar zijn met de text die ik je geef.
    - de voorbeeldtext wordt voorafgegaan door de term VOORBEELDTEKST
    - in de voorbeeldtext worden 1 of meer concepten benoemd die verwijzen naar een persoon anders dan de patient, het is belangrijk
    dat deze concepten in de nieuwe zin ook verwijzen naar iemand anders dan de patient (zoals een familielid), 
    het mogen ook andere medische concepten zijn.
    - de concepten die je moet vervangen zijn aangegeven met verticale streepjes, dus |concept|.
    - Een voorbeeld van een concept wat verwijst naar een ander persoon dan de patient =
    'Een zusje van #Name# is elders operatief behandeld in verband met recidiverende patella luxaties', waarin 'luxaties' het concept is, en er 
    wordt verwezen naar de zus van de patient.    
    - het domein is medisch dus gebruik medische concepten.
    - probeer de medische concepten te varieren, dus gebruik niet steeds dezelfde concepten.
    - varieer de ziektebeelden
    - varieer de opmaak van de text, dus gebruik niet steeds dezelfde opmaak.
    - geef als antwoord ALLEEN de nieuw gegenereerde text, voorafgaand met de term NIEUWE_TEKST
    - in de NIEUWE_TEKST, plaats alleen de concepten die verwijzen naar een ander persoon dan de patient tussen tussen verticale streepjes |, 
    dus bijvoorbeeld: 'Een zusje van #Name# is elders operatief behandeld in verband met recidiverende patella |luxaties|'
"""

SYSTEM_PROMPT_EXPERIENCER_CHECK = """
    Je bent een kritische assistent die mij helpt om nieuwe text te beoordelen.
    
    Je krijgt een tekst. Deze tekst bevatten 1 of meerdere concepten die zijn omsloten met verticale streepjes, dus |concept|.
    
    Het is jouw taak om te beoordelen of de concepten in de tekst verwijzen naar een persoon anders dan de patient (zoals een familielid, of een behandelend arts).
    LET OP: het gaat in de tekst om de verwijzing naar een persoon ANDERS dan de patient.
    LET OP: de tekst als geheel heeft betrekking op de patient.
    LET OP: het kan per concept verschillen of het verwijst naar een persoon anders dan de patient.
        
    De output die je geeft is beperkt tot 'ja' of 'nee' per concept, en wordt gegeven in de vorm van een dictionary:
    {0: 'ja', 1: 'nee', ...} 
    
    Hierin is 0, 1, ... de index van de concepten in de tekst.
    Wat betreft de index, begin altijd met 0, en tel op voor elk concept.

"""

SYSTEM_PROMPT_HISTORICAL = """
    Je bent een kritische assistent die mij helpt om nieuwe text te bedenken.
    Deze text moeten voldoen aan de volgende eisen:
    - het moet semantisch correct zijn en vergelijkbaar zijn met de text die ik je geef.
    - de voorbeeldtext wordt voorafgegaan door de term VOORBEELDTEKST
    - in de voorbeeldtext worden 1 of meer concepten benoemd die verwijzen naar een historische gebeurtenis, het is belangrijk
    dat deze concepten in de nieuwe zin ook verwijzen naar een historische gebeurtenis, het mogen ook andere medische concepten zijn.
    - de concepten die je moet vervangen zijn aangegeven met verticale streepjes, dus |concept|.
    - Een voorbeeld van een concept wat verwijst naar een historische gebeurtenis =
    'Een zusje van #Name# is 2 jaar geleden |operatief behandeld|', waarin 'operatief behandeld' het concept is, en er 
    wordt verwezen naar de zus van de patient.    
    - het domein is medisch, dus gebruik medische concepten.
    - probeer de medische concepten te varieren, dus gebruik niet steeds dezelfde concepten.
    - varieer de ziektebeelden
    - varieer de opmaak van de text, dus gebruik niet steeds dezelfde opmaak.
    - geef als antwoord ALLEEN de nieuw gegenereerde text, voorafgaand met de term NIEUWE_TEKST
    - in de NIEUWE_TEKST, plaats alleen de concepten die verwijzen naar een historische gebeurtenis tussen vertical streepjes |, 
    dus bijvoorbeeld: 'De |tumor| is vorig jaar verwijderd'
"""

SYSTEM_PROMPT_HISTORICAL_CHECK = """
    Je bent een kritische assistent die mij helpt om nieuwe text te beoordelen.
    
    Je krijgt een tekst. Deze tekst bevatten 1 of meerdere concepten die zijn omsloten met verticale streepjes, dus |concept|.
    
    Het is jouw taak om te beoordelen of deze concepten hebben plaatsgevonden in het verleden.
    LET OP: het gaat in de tekst om de verwijzing naar een concept in de verleden tijd.
    LET OP: de tekst als geheel heeft betrekking op de patient.
    LET OP: het kan per concept verschillen of het concept verwijst naar een recente of historische gebeurtenis.
        
    De output die je geeft is beperkt tot 'ja' of 'nee' per concept, en wordt gegeven in de vorm van een dictionary:
    {0: 'ja', 1: 'nee', ...} 
    
    Hierin is 0, 1, ... de index van de concepten in de tekst.
    Wat betreft de index, begin altijd met 0, en tel op voor elk concept.
"""